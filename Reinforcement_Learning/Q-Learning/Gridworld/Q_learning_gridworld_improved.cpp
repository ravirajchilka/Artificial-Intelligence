#include <torch/torch.h>
#include <iostream>
#include <random>
#include <algorithm>

constexpr int GRID_SIZE = 4;
constexpr int NUM_ACTIONS = 4;
constexpr int NUM_EPISODES = 1000;
constexpr float LEARNING_RATE = 0.1;
constexpr float DISCOUNT_FACTOR = 0.99;
constexpr float EPSILON = 0.1;
constexpr int MAX_STEPS = 50; // Prevent infinite loops

// Simple GridWorld environment
class GridWorld {
private:
    int agent_pos;
    int goal_pos;

public:
    GridWorld(int initial_agent_pos, int initial_goal_pos) :
        agent_pos(initial_agent_pos), goal_pos(initial_goal_pos) {}

    std::pair<int, float> step(int action) {
        int row = agent_pos / GRID_SIZE;
        int col = agent_pos % GRID_SIZE;

        switch (action) {
            case 0: row = std::max(0, row - 1); break; // Up
            case 1: row = std::min(GRID_SIZE - 1, row + 1); break; // Down
            case 2: col = std::max(0, col - 1); break; // Left
            case 3: col = std::min(GRID_SIZE - 1, col + 1); break; // Right
        }

        agent_pos = row * GRID_SIZE + col;
        float reward = (agent_pos == goal_pos) ? 1.0f : -0.1f;
        return {agent_pos, reward};
    }

    void reset(int start_pos) {
        agent_pos = start_pos;
    }

    bool is_terminal() const {
        return agent_pos == goal_pos;
    }
};

// Function to update Q-table
void update_q_table(torch::Tensor& q_table, int state, int action, float reward, int next_state) {
    float max_next_q = q_table.index({next_state}).max().item<float>();  
    float target = reward + DISCOUNT_FACTOR * max_next_q;  
    float current_q = q_table.index({state, action}).item<float>();  
    q_table.index_put_({state, action}, current_q + LEARNING_RATE * (target - current_q));
}

// Function to choose an action (exploration vs exploitation)
int choose_action(torch::Tensor& q_table, int state, std::mt19937& gen, float epsilon) {
    std::uniform_real_distribution<> dis(0.0, 1.0);
    if (dis(gen) < epsilon) {
        return std::uniform_int_distribution<>(0, NUM_ACTIONS - 1)(gen); // Random action
    }
    return q_table.index({state}).argmax().item<int>(); // Best action
}

int main() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> start_pos_dist(0, GRID_SIZE * GRID_SIZE - 1);

    // Initialize Q-table
    auto q_table = torch::zeros({GRID_SIZE * GRID_SIZE, NUM_ACTIONS});

    for (int episode = 0; episode < NUM_EPISODES; ++episode) {
        int start_pos = start_pos_dist(gen); // Random starting position
        GridWorld env(start_pos, GRID_SIZE * GRID_SIZE - 1);
        env.reset(start_pos);
        int state = start_pos;
        float total_reward = 0;
        int steps = 0;

        float epsilon = std::max(0.01f, EPSILON * std::exp(-0.001f * episode)); // Decay epsilon

        while (!env.is_terminal() && steps < MAX_STEPS) {
            steps++;
            int action = choose_action(q_table, state, gen, epsilon);
            auto [next_state, reward] = env.step(action);
            total_reward += reward;
            update_q_table(q_table, state, action, reward, next_state);
            state = next_state;
        }

        if (episode % 100 == 0) {
            std::cout << "Episode " << episode << ", Total Reward: " << total_reward << std::endl;
        }
    }

    // Print the final Q-table
    std::cout << "Final Q-table:" << std::endl;
    for (int i = 0; i < GRID_SIZE * GRID_SIZE; ++i) {
        std::cout << "State " << i << ": ";
        for (int j = 0; j < NUM_ACTIONS; ++j) {
            std::cout << q_table.index({i, j}).item<float>() << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
