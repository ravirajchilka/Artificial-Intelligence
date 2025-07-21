#include <torch/torch.h>
#include <iostream>
#include <random>
#include <algorithm>
#include <fstream>
#include <vector>

constexpr std::int16_t GRIDWORLD_SIZE = 4;
constexpr std::int16_t NUM_ACTIONS = 4;
constexpr std::uint16_t NUM_EPISODES = 2000;
constexpr std::int16_t MAX_STEPS_SIZE = 50;
constexpr float LEARNING_RATE = 0.01F;
constexpr float EPSILON = 0.1F;
constexpr float DISCOUNT_FACTOR = 0.99F;

struct QNetworkImpl : torch::nn::Module {
    torch::nn::Linear input_layer{nullptr};
    torch::nn::Linear hidden_layer{nullptr};
    torch::nn::Linear output_layer{nullptr};

    QNetworkImpl(int input_dim, int hidden_dim, int output_dim) {
        input_layer = register_module("input_layer", torch::nn::Linear(input_dim, hidden_dim));
        hidden_layer = register_module("hidden_layer", torch::nn::Linear(hidden_dim, hidden_dim));
        output_layer = register_module("output_layer", torch::nn::Linear(hidden_dim, output_dim));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(input_layer(x));
        x = torch::relu(hidden_layer(x));
        return output_layer(x);
    }
};
TORCH_MODULE(QNetwork);

class GridWorld {
public:
    GridWorld(std::int16_t agent_init_pos, std::int16_t goal_pos)
        : agent_pos(agent_init_pos), goal_pos(goal_pos) {}

    std::pair<std::int16_t, float> get_reward_state_pair(std::int16_t action) {
        int row = agent_pos / GRIDWORLD_SIZE;
        int col = agent_pos % GRIDWORLD_SIZE;

        switch (action) {
            case 0: row = std::max(0, row - 1); break;     // Up
            case 1: row = std::min(GRIDWORLD_SIZE - 1, row + 1); break; // Down
            case 2: col = std::max(0, col - 1); break;     // Left
            case 3: col = std::min(GRIDWORLD_SIZE - 1, col + 1); break; // Right
        }
        agent_pos = row * GRIDWORLD_SIZE + col;
        float reward = agent_pos == goal_pos ? 1.0F : -0.1F;
        return {agent_pos, reward};
    }

    bool isTerminal() const {
        return agent_pos == goal_pos;
    }

    void resetAgentPos(std::int16_t random_start_pos) {
        agent_pos = random_start_pos;
    }

private:
    std::int16_t agent_pos;
    std::int16_t goal_pos;
};

std::int16_t select_action(QNetwork& model, std::mt19937& gen, float epsilon, std::int16_t state) {
    static auto real_dis = std::uniform_real_distribution<float>(0.0F, 1.0F);
    static auto int_dis = std::uniform_int_distribution<int>(0, NUM_ACTIONS - 1);

    if (real_dis(gen) < epsilon) {
        return int_dis(gen);
    } else {
        torch::Tensor state_tensor = torch::tensor({static_cast<float>(state) / (GRIDWORLD_SIZE * GRIDWORLD_SIZE)}, torch::kFloat32).unsqueeze(0);
        torch::Tensor q_values = model->forward(state_tensor);
        return q_values.argmax(1).item<int>();
    }
}

int main() {
    QNetwork model(1, 64, NUM_ACTIONS);
    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(LEARNING_RATE));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> int_dist(0, (GRIDWORLD_SIZE * GRIDWORLD_SIZE) - 1);

    std::vector<float> episode_rewards;
    std::vector<float> episode_losses;

    for (std::int16_t episode = 0; episode < NUM_EPISODES; ++episode) {
        auto agent_initial_pos = int_dist(gen);
        GridWorld gridworld(agent_initial_pos, (GRIDWORLD_SIZE * GRIDWORLD_SIZE) - 1);
        gridworld.resetAgentPos(agent_initial_pos);

        std::int16_t current_state = agent_initial_pos;
        float total_reward = 0.0f;
        float total_loss = 0.0f;
        int step_count = 0;

        float epsilon = std::max(0.01F, EPSILON * std::exp(-0.003F * episode));

        while (!gridworld.isTerminal() && step_count++ < MAX_STEPS_SIZE) {
            std::int16_t action = select_action(model, gen, epsilon, current_state);
            auto [next_state, reward] = gridworld.get_reward_state_pair(action);
            total_reward += reward;

            torch::Tensor state_tensor = torch::tensor({static_cast<float>(current_state) / (GRIDWORLD_SIZE * GRIDWORLD_SIZE)}, torch::kFloat32).unsqueeze(0);
            torch::Tensor next_state_tensor = torch::tensor({static_cast<float>(next_state) / (GRIDWORLD_SIZE * GRIDWORLD_SIZE)}, torch::kFloat32).unsqueeze(0);

            torch::Tensor q_values = model->forward(state_tensor);
            torch::Tensor next_q_values = model->forward(next_state_tensor);
            float max_next_q = std::get<0>(next_q_values.max(1)).item<float>();
            float target_q = reward + DISCOUNT_FACTOR * max_next_q;

            torch::Tensor target = q_values.clone().detach();
            target[0][action] = target_q;

            torch::Tensor loss = torch::mse_loss(q_values, target);
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            total_loss += loss.item<float>();
            current_state = next_state;
        }

        float avg_loss = step_count > 0 ? total_loss / step_count : 0.0f;
        episode_rewards.push_back(total_reward);
        episode_losses.push_back(avg_loss);

        if (episode % 100 == 0) {
            std::cout << "Episode " << episode << " Total Reward: " << total_reward << std::endl;
        }
    }

    // Save metrics to CSV
    std::ofstream log_file("training_metrics.csv");
    log_file << "Episode,TotalReward,AverageLoss\n";
    for (size_t i = 0; i < episode_rewards.size(); ++i) {
        log_file << i << "," << episode_rewards[i] << "," << episode_losses[i] << "\n";
    }
    log_file.close();

    // Print final Q-values
    std::cout << "\nFinal NN-based Q-values for each state:\n";
    for (int s = 0; s < GRIDWORLD_SIZE * GRIDWORLD_SIZE; ++s) {
        torch::Tensor state_tensor = torch::tensor({static_cast<float>(s) / (GRIDWORLD_SIZE * GRIDWORLD_SIZE)}, torch::kFloat32).unsqueeze(0);
        torch::Tensor q_vals = model->forward(state_tensor);
        std::cout << "State " << s << ": ";
        for (int a = 0; a < NUM_ACTIONS; ++a) {
            std::cout << q_vals[0][a].item<float>() << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
