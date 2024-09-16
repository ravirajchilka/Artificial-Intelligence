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

// Simple grid world environment
class GridWorld {  
      private:
          int agent_pos;
          int goal_pos;

      public:
          //GridWorld() : agent_pos(0), goal_pos(GRID_SIZE * GRID_SIZE - 1) {}
          
          GridWorld(int initial_agent_pos, int initial_goal_pos) :
          agent_pos(initial_agent_pos), goal_pos(initial_goal_pos) {
              
          }

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

    //GridWorld env;
    GridWorld env (0,GRID_SIZE*GRID_SIZE-1);

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

            // Q-learning update using bellman equation Q(s,a) = Q(s,a) + alpha * [R + gamma * max(Q(s',a')) - Q(s,a)]
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

    // Print the final Q-table
    std::cout << "Final Q-table:" << std::endl;
    for (int i = 0; i < GRID_SIZE * GRID_SIZE; ++i) {
        std::cout << "State " << i << ": ";
        for (int j = 0; j < NUM_ACTIONS; ++j) {
            std::cout << q_table[i][j].item<float>() << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}

