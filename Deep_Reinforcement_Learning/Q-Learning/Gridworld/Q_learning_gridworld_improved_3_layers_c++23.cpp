#include <torch/torch.h>
#include <random>
#include <algorithm>
#include <print>
#include <expected>
#include <generator>
#include <ranges>
#include <cmath>

constexpr std::int16_t GRIDWORLD_SIZE = 4;
constexpr std::int16_t NUM_ACTIONS = 4;
constexpr std::uint16_t NUM_EPISODES = 2000;
constexpr std::int16_t MAX_STEPS_SIZE = 50;
constexpr float LEARNING_RATE = 0.01F;
constexpr float EPSILON = 0.1F;
constexpr float DISCOUNT_FACTOR = 0.99F;

// Compile-time normalization function
constexpr float normalize_state(int s) {
    return static_cast<float>(s) / (GRIDWORLD_SIZE * GRIDWORLD_SIZE);
}

struct QNetworkImpl : torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};  // Three layers

    QNetworkImpl(int input_dim, int output_dim) {
        fc1 = register_module("fc1", torch::nn::Linear(input_dim, 64));
        fc2 = register_module("fc2", torch::nn::Linear(64, 64));  // Additional hidden layer
        fc3 = register_module("fc3", torch::nn::Linear(64, output_dim));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1(x));  // Apply ReLU after first layer
        x = torch::relu(fc2(x));  // Apply ReLU after second layer
        return fc3(x);  // Output Q-values for all actions after third layer
    }
};
TORCH_MODULE(QNetwork);

class GridWorld {
	public:
		GridWorld(std::int16_t agent_init_pos, std::int16_t goal_pos)
			: agent_pos(agent_init_pos), goal_pos(goal_pos) {}

		using StateReward_c = std::pair<std::int16_t, float>;
		StateReward_c get_reward_state_pair(std::int16_t action) {
			auto [row, col] = std::pair{agent_pos / GRIDWORLD_SIZE, agent_pos % GRIDWORLD_SIZE};
			switch (action) {
				case 0: row = std::max(0, row - 1); break; // Up
				case 1: row = std::min(GRIDWORLD_SIZE - 1, row + 1); break; // Down
				case 2: col = std::max(0, col - 1); break; // Left
				case 3: col = std::min(GRIDWORLD_SIZE - 1, col + 1); break; // Right
			}
			agent_pos = row * GRIDWORLD_SIZE + col;
			float reward = agent_pos == goal_pos ? 1.0F : -0.1F;
			return {agent_pos, reward};
		}

		bool isTerminal() const { return agent_pos == goal_pos; }
		void resetAgentPos(std::int16_t pos) { agent_pos = pos; }

	private:
		std::int16_t agent_pos;
		std::int16_t goal_pos;
};

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

// C++23: Use structured bindings and concise lambda
std::int16_t select_action(QNetwork& model, std::mt19937& gen, float epsilon, std::int16_t state) {
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

// C++23: Use std::generator for episode steps
auto episode_steps(GridWorld& env, QNetwork& model, std::mt19937& gen, float epsilon, std::int16_t start_state)
    -> std::generator<std::tuple<std::int16_t, std::int16_t, float, std::int16_t>>
{
    std::int16_t state = start_state;
    std::int16_t steps = 0;
    while (!env.isTerminal() && steps++ < MAX_STEPS_SIZE) {
        std::int16_t action = select_action(model, gen, epsilon, state);
        auto [next_state, reward] = env.get_reward_state_pair(action);
        co_yield {state, action, reward, next_state};
        state = next_state;
    }
}

int main() {
    auto model_result = create_model();
    if (!model_result) {
        std::println("{}", model_result.error());
        return -1;
    }
    auto& model = model_result.value();

    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(LEARNING_RATE));
    std::random_device rd;
    std::mt19937 gen(rd());
    auto int_dist = std::uniform_int_distribution<int>(0, (GRIDWORLD_SIZE * GRIDWORLD_SIZE) - 1);

    for (std::int16_t episode = 0; episode < NUM_EPISODES; ++episode) {
        auto agent_initial_pos = int_dist(gen);
        GridWorld gridworld(agent_initial_pos, (GRIDWORLD_SIZE * GRIDWORLD_SIZE) - 1);
        gridworld.resetAgentPos(agent_initial_pos);

        float total_reward = 0.0F;
        float epsilon = std::lerp(EPSILON, 0.01F, static_cast<float>(episode) / NUM_EPISODES);

        for (auto [state, action, reward, next_state] : episode_steps(gridworld, model, gen, epsilon, agent_initial_pos)) {
            total_reward += reward;

            torch::Tensor state_tensor = torch::tensor({normalize_state(state)}, torch::kFloat32).unsqueeze(0);
            torch::Tensor next_state_tensor = torch::tensor({normalize_state(next_state)}, torch::kFloat32).unsqueeze(0);

            torch::Tensor q_values = model->forward(state_tensor);
            torch::Tensor next_q_values = model->forward(next_state_tensor);

            // C++23: Use std::ranges for max Q-value
            auto q_range = std::ranges::subrange(next_q_values[0].data_ptr<float>(), next_q_values[0].data_ptr<float>() + NUM_ACTIONS);
            float target_q = reward + DISCOUNT_FACTOR * std::ranges::max(q_range);

            torch::Tensor target = q_values.clone().detach();
            target[0][action] = target_q;

            torch::Tensor loss = torch::mse_loss(q_values, target);
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
        }

        if (episode % 100 == 0) {
            std::println("Episode {} -> Total Reward: {:.2f}", episode, total_reward);
        }
    }

    std::println("\nFinal Q-values learned by the neural network:");
    for (int s : std::views::iota(0, GRIDWORLD_SIZE * GRIDWORLD_SIZE)) {
        torch::Tensor state_tensor = torch::tensor({normalize_state(s)}, torch::kFloat32).unsqueeze(0);
        torch::Tensor q_vals = model->forward(state_tensor);

        std::print("State {}: ", s);
        for (int a = 0; a < NUM_ACTIONS; ++a) {
            std::print("{:.3f} ", q_vals[0][a].item<float>());
        }
        std::print("\n");
    }

    return 0;
}
