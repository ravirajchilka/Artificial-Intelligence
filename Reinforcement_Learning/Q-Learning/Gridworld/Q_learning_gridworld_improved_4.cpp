#include <torch/torch.h>
#include <iostream>
#include <cstdint>
#include <algorithm>
#include <random>

constexpr std::int16_t GRIDWORLD_SIZE = 4;
constexpr std::int16_t NUM_ACTIONS = 4;
constexpr std::uint16_t NUM_EPISODES = 2000;
constexpr std::int16_t MAX_STEPS_SIZE = 50;
constexpr float LEARNING_RATE = 0.1F;
constexpr float EPSILON = 0.1F;
constexpr float DISCOUNT_FACTOR = 0.99F;

class GridWorld {
    public:
        GridWorld(std::int16_t agent_init_pos, std::int16_t goal_pos)
            : agent_pos(agent_init_pos), goal_pos(goal_pos) {}

        std::pair<std::int16_t, float> get_reward_state_pair(std::int16_t action) {
            int row = agent_pos / GRIDWORLD_SIZE;
            int col = agent_pos % GRIDWORLD_SIZE;

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

        bool isTerminal() {
            return agent_pos == goal_pos;
        }

        void resetAgentPos(std::int16_t random_start_pos) {
            agent_pos = random_start_pos;
        }

    private:
        std::int16_t agent_pos;
        std::int16_t goal_pos;
};

void build_q_table(torch::Tensor &q_table, std::int16_t current_state, std::int16_t next_state, float reward, std::int16_t action) {
    float max_next_state_q = q_table[next_state].max().item<float>();
    float target = reward + max_next_state_q * DISCOUNT_FACTOR;
    float current_q = q_table.index({current_state,action}).item<float>();
    float temporal_difference = target - current_q;
    float final_q_value = current_q + LEARNING_RATE * temporal_difference;
    q_table.index_put_({current_state,action}, final_q_value);
}

std::int16_t action_exploration_exploitation(torch::Tensor &q_table ,std::mt19937 &gen, float epsilon, std::int16_t current_state) {
    static auto real_dis = std::uniform_real_distribution<float>(0.0F,1.0F);
    static auto int_dis = std::uniform_int_distribution<int>(0,NUM_ACTIONS-1);
    if(real_dis(gen) < epsilon) {
        return int_dis(gen);
    } else {
        return q_table[current_state].argmax().item<int>();
    }
}


int main() {
    auto q_table = torch::zeros({(GRIDWORLD_SIZE * GRIDWORLD_SIZE),NUM_ACTIONS});
    std::random_device rd;
    std::mt19937 gen(rd());
    auto int_dist = std::uniform_int_distribution<int>(0,(GRIDWORLD_SIZE * GRIDWORLD_SIZE) - 1);
    
    for(std::int16_t episode = 0; episode < NUM_EPISODES; ++episode) {
        auto agent_initial_pos = int_dist(gen);
        GridWorld gridworld(agent_initial_pos,(GRIDWORLD_SIZE * GRIDWORLD_SIZE) - 1);
        gridworld.resetAgentPos(agent_initial_pos);
        
        std::int16_t current_state = agent_initial_pos;
        float total_reward = 0;
        std::int16_t steps = 0;

        float epsilon = std::max(0.01F, EPSILON * std::exp(-0.003F * episode));

        while(!gridworld.isTerminal()) {
            steps++;
            std::int16_t action = action_exploration_exploitation(q_table,gen,epsilon,current_state);
            auto [next_state,reward] = gridworld.get_reward_state_pair(action);
            total_reward += reward;
            build_q_table(q_table,current_state,next_state,reward,action);
            current_state = next_state;
        }

        if(episode % 100 == 0) {
            std::cout << "episode no. " << episode << "total reward " << total_reward << std::endl;
        }
    }


       // Print the final Q-table
       std::cout << "Final Q-table:" << std::endl;
       for (int i = 0; i < GRIDWORLD_SIZE * GRIDWORLD_SIZE; ++i) {
           std::cout << "State " << i << ": ";
           for (int j = 0; j < NUM_ACTIONS; ++j) {
               std::cout << q_table.index({i, j}).item<float>() << " ";
           }
           std::cout << std::endl;
       }   


    return 0;
}


