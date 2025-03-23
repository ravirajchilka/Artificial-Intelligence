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

			  // Hard restriction: If danger zone, stay in the same place
			  if (agent_pos == 9 || agent_pos == 10) {
				return {agent_pos, -100.0F}; // assigning very high negative value
			}
            
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

/*
	Sample Final Q-table:
	State 0: -0.129223 0.460871 -0.135593 -0.105162      
	State 1: -0.0896408 0.566553 -0.105995 0.0158365     
	State 2: -0.0797109 0.673275 -0.0441048 -0.0726085   
	State 3: -0.03994 0.781096 -0.0421424 0.0218113      
	State 4: -0.119556 -0.047648 -0.00106291 0.566555    
	State 5: -0.0342518 -34.25 0.0561901 0.673288        
	State 6: 0.105434 -34.1627 0.193996 0.781099
	State 7: 0.193529 0.89 0.110067 0.199785
	State 8: -0.0933192 0.673282 -0.0893053 -10
	State 9: -0.020088 0.781096 -0.0279864 -10
	State 10: 0.0583287 0.1603 -26.966 0.889999
	State 11: 0.474605 1 -34.1346 0.364452
	State 12: 0.0560295 -0.0499001 -0.0499001 0.781099   
	State 13: -34.2093 -0.01999 0.121812 0.89 
	State 14: -40.7593 0.231257 0.213295 1
	State 15: 0 0 0 0
*/

