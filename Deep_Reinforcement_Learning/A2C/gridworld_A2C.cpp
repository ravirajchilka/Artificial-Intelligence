#include <torch/torch.h>
#include <raylib.h>
#include <iostream>
#include <cstdint>
#include <random>
#include <cmath>
#include <expected>
#include <ranges>
#include <print>

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

// --- A2C Neural Network Classes ---
struct ActorNetworkImpl : torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
    ActorNetworkImpl(int input_dim, int output_dim) {
        fc1 = register_module("fc1", torch::nn::Linear(input_dim, 64));
        fc2 = register_module("fc2", torch::nn::Linear(64, 64));
        fc3 = register_module("fc3", torch::nn::Linear(64, output_dim));
    }
    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1(x));
        x = torch::relu(fc2(x));
        return fc3(x); // Output raw policy logits (no softmax)
    }
};

TORCH_MODULE(ActorNetwork);

struct CriticNetworkImpl : torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
    CriticNetworkImpl(int input_dim) {
        fc1 = register_module("fc1", torch::nn::Linear(input_dim, 64));
        fc2 = register_module("fc2", torch::nn::Linear(64, 64));
        fc3 = register_module("fc3", torch::nn::Linear(64, 1));
    }
    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1(x));
        x = torch::relu(fc2(x));
        return fc3(x); // Output state value
    }
};

TORCH_MODULE(CriticNetwork);

// --- State Encoding for NN ---
inline int pos_to_state(Vector2 pos) {
    int x = static_cast<int>((pos.x - 100) / 200);
    int y = static_cast<int>((pos.y - 100) / 200);
    return y * GRID_SIZE + x;
}

// --- NN Action Selection ---
std::int16_t select_action(ActorNetwork& actor_model, std::mt19937& gen, float epsilon, int current_state) {
    static auto real_dis = std::uniform_real_distribution<float>(0.0F, 1.0F);
    static auto int_dis = std::uniform_int_distribution<int>(0, NUM_ACTIONS - 1);
    if (real_dis(gen) < epsilon) {
        return int_dis(gen);
    } else {
        torch::Tensor state_tensor = torch::tensor({static_cast<float>(current_state)}, torch::kFloat32).unsqueeze(0);
        torch::Tensor logits = actor_model->forward(state_tensor);
        torch::Tensor action_probs = torch::softmax(logits, 1);
        return action_probs.argmax(1).item<int>();
    }
}

// --- A2C Training Step ---
void build_a2c(ActorNetwork& actor_model, CriticNetwork& critic_model,
               torch::optim::Adam& actor_optimizer, torch::optim::Adam& critic_optimizer,
               int state, int action, float reward, int next_state, bool done) {
    torch::Tensor state_tensor = torch::tensor({static_cast<float>(state)}, torch::kFloat32).unsqueeze(0);
    torch::Tensor next_state_tensor = torch::tensor({static_cast<float>(next_state)}, torch::kFloat32).unsqueeze(0);

    // Compute critic value
    torch::Tensor value = critic_model->forward(state_tensor);
    torch::Tensor next_value = critic_model->forward(next_state_tensor);
    
    // TD Target
    float target_value = reward + (done ? 0.0f : DISCOUNT_FACTOR * next_value.item<float>());
    
    // Critic loss (MSE)
    torch::Tensor critic_loss = torch::mse_loss(value, torch::tensor(target_value, torch::kFloat32));

    // Advantage
    float advantage = target_value - value.item<float>();

    // Actor loss (Negative log probability * advantage)
    torch::Tensor logits = actor_model->forward(state_tensor);
    torch::Tensor action_probs = torch::softmax(logits, 1);
    torch::Tensor log_prob = torch::log(action_probs[0][action]);
    torch::Tensor actor_loss = -log_prob * advantage;

    // Update networks
    critic_optimizer.zero_grad();
    critic_loss.backward();
    critic_optimizer.step();

    actor_optimizer.zero_grad();
    actor_loss.backward();
    actor_optimizer.step();
}

int main() {
    ActorNetwork actor_model(1, NUM_ACTIONS);
    CriticNetwork critic_model(1);

    torch::optim::Adam actor_optimizer(actor_model->parameters(), torch::optim::AdamOptions(LEARNING_RATE));
    torch::optim::Adam critic_optimizer(critic_model->parameters(), torch::optim::AdamOptions(LEARNING_RATE));

    InitWindow(800, 800, "Gridworld A2C");
    SetTargetFPS(60);
    Grids grid(GRID_SIZE, GRID_SIZE);

    std::random_device rd;
    std::mt19937 gen(rd());
    auto start_position_dist = std::uniform_int_distribution<int>(0, GRID_SIZE - 1);

    int episode = 1;
    int start_x = start_position_dist(gen);
    int start_y = start_position_dist(gen);
    while (start_x == GRID_SIZE - 1 && start_y == GRID_SIZE - 1) {
        start_x = start_position_dist(gen);
        start_y = start_position_dist(gen);
    }

    Ball agent(Vector2{static_cast<float>(start_x * 200 + 100),
                       static_cast<float>(start_y * 200 + 100)} ,
               Vector2{700.0f, 700.0f});

    float total_reward = 0.0f;
    float epsilon = std::lerp(EPSILON, 0.01F, static_cast<float>(episode) / NUM_EPISODES);

    // Main training loop
    while (!WindowShouldClose()) {
        BeginDrawing();
        ClearBackground(RAYWHITE);

        grid.draw_grid();
        grid.draw_goal_state();
        agent.draw_ball();

        // A2C training step
        int state = pos_to_state(agent.get_position());
        int action = select_action(actor_model, gen, epsilon, state);
        auto [next_pos, reward] = agent.get_reward_state_pair(action);
        int next_state = pos_to_state(next_pos);
        bool done = agent.is_terminal_state();

        build_a2c(actor_model, critic_model, actor_optimizer, critic_optimizer, state, action, reward, next_state, done);

        total_reward += reward;

        // Render text and update state
        if (done) {
            std::cout << "Episode: " << episode << " | Reward: " << total_reward << '\n';
            episode++;
            if (episode > NUM_EPISODES) break;
            // Reset agent for next episode
            start_x = start_position_dist(gen);
            start_y = start_position_dist(gen);
            while (start_x == GRID_SIZE - 1 && start_y == GRID_SIZE - 1) {
                start_x = start_position_dist(gen);
                start_y = start_position_dist(gen);
            }
            agent = Ball(Vector2{static_cast<float>(start_x * 200 + 100),
                                 static_cast<float>(start_y * 200 + 100)}, Vector2{700.0f, 700.0f});
            total_reward = 0.0f;
        }

        EndDrawing();
    }

    CloseWindow();  // Close window and OpenGL context
    return 0;
}
