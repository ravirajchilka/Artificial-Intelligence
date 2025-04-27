#include <torch/torch.h>
#include <raylib.h>
#include <iostream>
#include <cstdint>
#include <random>
#include <cmath>
#include <expected>
#include <print>
#include <ranges>
#include <generator>

constexpr std::int16_t GRIDWORLD_SIZE = 4;
constexpr std::int16_t NUM_ACTIONS = 4;
constexpr std::uint16_t NUM_EPISODES = 500;
constexpr float LEARNING_RATE = 0.01F;
constexpr float EPSILON = 0.1F;
constexpr float DISCOUNT_FACTOR = 0.99F;

// --- Raylib Environment Classes ---
constexpr std::int16_t GRID_SIZE = 4; // 4x4 grid

class Grids {
	public:
		Grids(std::int8_t rows_val, std::int8_t cols_val) : rows(rows_val), cols(cols_val) {}
		void draw_grid() const {
			DrawLineEx({0.0f, 0.0f}, {0.0f, 800.0f}, 5.0f, BLACK); // vertical left
			DrawLineEx({0.0f, 0.0f}, {800.0f, 0.0f}, 5.0f, BLACK); // horizontal top
			DrawLineEx({800.0f, 0.0f}, {800.0f, 800.0f}, 5.0f, BLACK); // vertical right
			DrawLineEx({0.0f, 800.0f}, {800.0f, 800.0f}, 5.0f, BLACK); // horizontal bottom
			for (auto i : std::views::iota(1, GRID_SIZE)) {
				DrawLineEx({0.0f, static_cast<float>(200 * i)},
						{800.0f, static_cast<float>(200 * i)}, 5.0f, BLACK);
				DrawLineEx({static_cast<float>(i * 200), 0.0f},
						{static_cast<float>(i * 200), 800.0f}, 5.0f, BLACK);
			}
		}
		void draw_goal_state() const {
			DrawRectangle(600, 600, 200, 200, GREEN);
		}
	private:
		std::int8_t rows;
		std::int8_t cols;
};

class Ball {
	public:
		Ball(Vector2 agent_init_pos, Vector2 goal_pos)
				: ball_pos(agent_init_pos), goal_pos(goal_pos) {}
		void draw_ball() const {
			DrawCircleV(ball_pos, 40, RED);
		}
		void move_ball_forward() {
			if (ball_pos.x < 700) ball_pos.x += 200;
		}
		void move_ball_backward() {
			if (ball_pos.x > 100) ball_pos.x -= 200;
		}
		void move_ball_down() {
			if (ball_pos.y < 700) ball_pos.y += 200;
		}
		void move_ball_up() {
			if (ball_pos.y > 100) ball_pos.y -= 200;
		}
		bool is_terminal_state() const {
			return ball_pos.x == goal_pos.x && ball_pos.y == goal_pos.y;
		}
		
		Vector2 get_position() const { 
			return ball_pos; 
		}

		using StateReward_c_type = std::pair<Vector2, float>;
		StateReward_c_type get_reward_state_pair(std::int16_t action) {
			switch (action) {
				case 0: move_ball_forward(); break; // Right
				case 1: move_ball_backward(); break; // Left
				case 2: move_ball_up(); break; // Up
				case 3: move_ball_down(); break; // Down
			}
			float reward = is_terminal_state() ? 1.0F : -0.1F;
			return {ball_pos, reward};
		}
	private:
		Vector2 ball_pos;
		Vector2 goal_pos;
};


// --- DQN NN Classes ---
constexpr float normalize_state(int s) {
    return static_cast<float>(s) / (GRIDWORLD_SIZE * GRIDWORLD_SIZE);
}

struct QNetworkImpl : torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
    QNetworkImpl(int input_dim, int output_dim) {
        fc1 = register_module("fc1", torch::nn::Linear(input_dim, 64));
        fc2 = register_module("fc2", torch::nn::Linear(64, 64));
        fc3 = register_module("fc3", torch::nn::Linear(64, output_dim));
    }
    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1(x));
        x = torch::relu(fc2(x));
        return fc3(x);
    }
};
TORCH_MODULE(QNetwork);

// C++23: Use std::expected for model creation
[[nodiscard]]
std::expected<QNetwork, std::string> create_model() {
    try {
        QNetwork model(1, NUM_ACTIONS);
        return model;
    } catch (const std::exception& e) {
        return std::unexpected(std::string("Failed to create model: ") + e.what());
    }
}

// --- State Encoding for NN ---
inline int pos_to_state(Vector2 pos) {
    int x = static_cast<int>((pos.x - 100) / 200);
    int y = static_cast<int>((pos.y - 100) / 200);
    return y * GRID_SIZE + x;
}

// --- NN Action Selection ---
std::int16_t select_action(QNetwork& model, std::mt19937& gen, float epsilon, int state) {
    static auto real_dis = std::uniform_real_distribution<float>(0.0F, 1.0F);
    static auto int_dis = std::uniform_int_distribution<int>(0, NUM_ACTIONS - 1);
    if (real_dis(gen) < epsilon) {
        return int_dis(gen);
    } else {
        torch::Tensor state_tensor = torch::tensor({normalize_state(state)}, torch::kFloat32).unsqueeze(0);
        torch::Tensor q_values = model->forward(state_tensor);
        return q_values.argmax(1).item<int>();
    }
}

// --- NN Training Step ---
void dqn_update(QNetwork& model, torch::optim::Adam& optimizer,int state, 
	int action, float reward, int next_state, bool done) {
    torch::Tensor state_tensor = torch::tensor({normalize_state(state)}, torch::kFloat32).unsqueeze(0);
    torch::Tensor next_state_tensor = torch::tensor({normalize_state(next_state)}, torch::kFloat32).unsqueeze(0);

    torch::Tensor q_values = model->forward(state_tensor);
    torch::Tensor next_q_values = model->forward(next_state_tensor);

    // C++23: Use std::ranges for max Q-value
    auto q_range = std::ranges::subrange(next_q_values[0].data_ptr<float>(), next_q_values[0].data_ptr<float>() + NUM_ACTIONS);
    float max_next_q = std::ranges::max(q_range);

    float target_q = reward + (done ? 0.0f : DISCOUNT_FACTOR * max_next_q);

    torch::Tensor target = q_values.clone().detach();
    target[0][action] = target_q;

    torch::Tensor loss = torch::mse_loss(q_values, target);
    optimizer.zero_grad();
    loss.backward();
    optimizer.step();
}

// --- C++23: Use std::generator for episode steps ---
auto episode_steps(Ball& agent, QNetwork& model, std::mt19937& gen, float epsilon)
    -> std::generator<std::tuple<int, int, float, int, bool>>
{
    while (!agent.is_terminal_state()) {
        int state = pos_to_state(agent.get_position());
        int action = select_action(model, gen, epsilon, state);
        auto [next_pos, reward] = agent.get_reward_state_pair(action);
        int next_state = pos_to_state(next_pos);
        bool done = agent.is_terminal_state();
        co_yield {state, action, reward, next_state, done};
        if (done) break;
    }
}

// --- Main Loop ---
int main() {
    auto model_result = create_model();
    if (!model_result) {
        std::println("{}", model_result.error());
        return -1;
    }
    auto& model = model_result.value();
    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(LEARNING_RATE));

    InitWindow(800, 800, "Gridworld DQN (C++23 + generator)");
    SetTargetFPS(60);
    Grids grid(GRID_SIZE, GRID_SIZE);

    std::random_device rd;
    std::mt19937 gen(rd());
    auto start_position_dist = std::uniform_int_distribution<int>(0, GRID_SIZE - 1);

    for (int episode : std::views::iota(1, NUM_EPISODES + 1)) {
        int start_x = start_position_dist(gen);
        int start_y = start_position_dist(gen);
        while (start_x == GRID_SIZE - 1 && start_y == GRID_SIZE - 1) {
            start_x = start_position_dist(gen);
            start_y = start_position_dist(gen);
        }

        Ball agent(Vector2{static_cast<float>(start_x * 200 + 100),
                           static_cast<float>(start_y * 200 + 100)},
                   Vector2{700.0f, 700.0f});

        float total_reward = 0.0f;
        float epsilon = std::lerp(EPSILON, 0.01F, static_cast<float>(episode) / NUM_EPISODES);

        for (auto [state, action, reward, next_state, done] : episode_steps(agent, model, gen, epsilon)) {
            total_reward += reward;
            dqn_update(model, optimizer, state, action, reward, next_state, done);

            BeginDrawing();
            ClearBackground(RAYWHITE);
            grid.draw_grid();
            grid.draw_goal_state();
            agent.draw_ball();
            EndDrawing();

            if (done) break;
        }
        if (episode % 50 == 0)
            std::println("Episode {} finished, Total Reward: {:.2f}", episode, total_reward);
    }

    std::println("\nFinal Q-values learned by the neural network:");
    for (int s : std::views::iota(0, GRIDWORLD_SIZE * GRIDWORLD_SIZE)) {
        torch::Tensor state_tensor = torch::tensor({normalize_state(s)}, torch::kFloat32).unsqueeze(0);
        torch::Tensor q_vals = model->forward(state_tensor);
        std::print("State {}: ", s);
        for (int a = 0; a < NUM_ACTIONS; ++a)
            std::print("{:.3f} ", q_vals[0][a].item<float>());
        std::print("\n");
    }

    CloseWindow();
    return 0;
}
