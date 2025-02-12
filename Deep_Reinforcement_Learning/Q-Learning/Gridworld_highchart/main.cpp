#include <torch/torch.h>
#include <iostream>
#include <random>
#include <algorithm>
#include <fstream>
#include "json.hpp"

constexpr int GRID_SIZE = 4;
constexpr int NUM_ACTIONS = 4;
constexpr int NUM_EPISODES = 1000;
constexpr float LEARNING_RATE = 0.1;
constexpr float DISCOUNT_FACTOR = 0.99;
constexpr float EPSILON = 0.1;

using json = nlohmann::json;

// Simple grid world environment
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

    void reset() {
        agent_pos = 0;
    }

    bool is_terminal() {
        return agent_pos == goal_pos;
    }
};

int main() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    // Initialize Q-table
    auto q_table = torch::zeros({GRID_SIZE * GRID_SIZE, NUM_ACTIONS});

    GridWorld env(0, GRID_SIZE * GRID_SIZE - 1);

    for (int episode = 0; episode < NUM_EPISODES; ++episode) {
        env.reset();
        int state = 0;
        float total_reward = 0;

        while (!env.is_terminal()) {
            int action;
            if (dis(gen) < EPSILON) {
                // Explore: choose a random action
                action = std::uniform_int_distribution<>(0, NUM_ACTIONS - 1)(gen);
            } else {
                // Exploit: choose the best action
                action = q_table[state].argmax().item<int>();
            }

            auto [next_state, reward] = env.step(action);
            total_reward += reward;

            // Q-learning update
            float max_next_q = q_table[next_state].max().item<float>();
            float target = reward + DISCOUNT_FACTOR * max_next_q;
            float current_q = q_table[state][action].item<float>();
            q_table[state][action] = current_q + LEARNING_RATE * (target - current_q);

            state = next_state;
        }

        if (episode % 100 == 0) {
            std::cout << "Episode " << episode << ", Total Reward: " << total_reward << std::endl;
        }
    }

    // Action labels for each action
    std::vector<std::string> actions = {"Up", "Down", "Left", "Right"};

    // Convert Q-table to JSON format with sorted keys
    json q_table_json;

    // Create a vector of key-value pairs
    std::vector<std::pair<int, json>> key_value_pairs;

    for (int i = 0; i < GRID_SIZE * GRID_SIZE; ++i) {
        json state_q_values;
        for (int j = 0; j < NUM_ACTIONS; ++j) {
            state_q_values.push_back(q_table[i][j].item<float>());
        }
        
        // Add the action labels along with the Q-values
        json action_labels;
        for (const auto& action : actions) {
            action_labels.push_back(action);
        }
        
        // Include action labels in the state data
        json state_data;
        state_data["q_values"] = state_q_values;
        state_data["actions"] = action_labels;
        
        key_value_pairs.emplace_back(i, state_data);
    }

    // Sort the key-value pairs by key (state index)
    std::sort(key_value_pairs.begin(), key_value_pairs.end(),
              [](const std::pair<int, json>& a, const std::pair<int, json>& b) {
                  return a.first < b.first;
              });

    // Insert sorted key-value pairs into the JSON object
    for (const auto& pair : key_value_pairs) {
        q_table_json[std::to_string(pair.first)] = pair.second;
    }

    // Save the JSON to a file
    std::ofstream file("q_table.json");
    if (file.is_open()) {
        file << q_table_json.dump(4); // Pretty print with indentation
        file.close();
        std::cout << "Q-table saved to q_table.json" << std::endl;
    } else {
        std::cerr << "Error saving Q-table to file!" << std::endl;
    }

    return 0;
}
