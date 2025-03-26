#include <torch/torch.h>
#include <iostream>
#include <random>
#include <cstdint>

constexpr std::int16_t NUM_STATES = 36;
constexpr std::int16_t NUM_MAX_STEPS = 20;
constexpr std::int16_t NUM_ACTIONS = 2;
constexpr std::int16_t NUM_EPISODES = 1000;
constexpr float LEARNING_RATE = 0.18F; // increased learning rate
constexpr float EPSILON = 0.1F;
constexpr float DISCOUNT_FACTOR = 0.99F;

class Pendulum {
    public : 
        Pendulum(std::int16_t bob_init_state,std::int16_t pendulum_goal_state)
		:bob_state(bob_init_state),pendulum_goal_state(pendulum_goal_state) {}

		std::pair<float,std::int16_t>return_reward_state(std::int16_t action) {
            
            bob_store_state = bob_state;
   
			switch(action) {
				case 0: bob_state += 10;
				break;	
				case 1: bob_state -= 10;	
				break;	
			}
			
			float reward = 0;
			std::int16_t state = (bob_state / 10) + 2; 
            // had to add 2 to properly map init state to current state
			
			if(bob_state > bob_store_state)	{
				reward = 1.0F;
			} else {
				reward = -0.1F;
			}

			if(bob_state < 30 || bob_state > 80) {
				reward = -1.0F; 
			}

			return {reward,state};
		}

		void randomly_init_bob_state(std::int16_t random_state) {
			bob_state = random_state;
		}

		bool is_terminal() {
			return bob_state == pendulum_goal_state;
		}

    private : 
		std::int16_t bob_state;
		std::int16_t bob_store_state;
		std::int16_t pendulum_goal_state;
};

void build_q_table(torch::Tensor &q_table, float reward, std::int16_t action, std::int16_t next_state, std::int16_t current_state) {
	float max_next_state_q = q_table.index({next_state}).max().item<float>(); 
	float current_q = q_table.index({current_state,action}).item<float>();
	float target = reward + max_next_state_q * DISCOUNT_FACTOR;
	float temporal_difference = target - current_q;
	float final_q = current_q + LEARNING_RATE * temporal_difference;
	q_table.index_put_({current_state,action},final_q);
}

std::int16_t return_action_exploration_exploitation(torch::Tensor &q_table, std::mt19937 &gen, float decaying_epsilon, std::int16_t current_state) {
	auto int_dist = std::uniform_int_distribution(0,NUM_ACTIONS-1);
	auto real_dist = std::uniform_real_distribution(0.0F,1.0F);
	if(decaying_epsilon > real_dist(gen)) {
		return int_dist(gen);
	} else {
		return q_table.index({current_state}).argmax().item<int>();
	}
}

int main() {
	torch::Tensor q_table = torch::zeros({NUM_STATES,NUM_ACTIONS});
	std::random_device rd;
	std::mt19937 gen(rd());

	for(std::int16_t episode = 0; episode < NUM_EPISODES; ++episode)  {
		auto int_dis = std::uniform_int_distribution(3,7);
		std::int16_t current_state = int_dis(gen);
        //std::int16_t current_state = 3;
		std::int16_t terminal_state = 80;
		Pendulum pendulum(current_state * 10,terminal_state);
		float total_reward = 0;
		std::int16_t steps = 0;
		float decaying_epsilon = std::max(0.01F,EPSILON * std::exp(-0.003F * episode));

		while(!pendulum.is_terminal() && steps < NUM_MAX_STEPS) {
			steps++;
			std::int16_t action = return_action_exploration_exploitation(q_table,gen,decaying_epsilon,current_state); 
			auto [reward,next_state] = pendulum.return_reward_state(action);
			total_reward += reward;
			build_q_table(q_table,reward,action,next_state,current_state);
			current_state = next_state;
		}
		
		if(episode % 100 == 0) {
            std::cout << "episode no. " << episode << "total reward " << total_reward << std::endl;
        }
	}

	  // Print the final Q-table
       std::cout << "Final Q-table:" << std::endl;
       for (int i = 0; i < NUM_STATES; ++i) {
           std::cout << "State " << i << ": ";
           for (int j = 0; j < NUM_ACTIONS; ++j) {
               std::cout << q_table.index({i, j}).item<float>() << " ";
           }
           std::cout << std::endl;
       }   
	return 0;   

}
