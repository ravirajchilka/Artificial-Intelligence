#include <iostream>
#include <random>
#include <torch/torch.h>

constexpr int NUM_ACTIONS = 2;         // Only one action: increase angle
constexpr int NUM_STATES = 6;          // 6 possible states (angles 30, 40, ..., 80)
constexpr float LEARNING_RATE = 0.1f;
constexpr float DISCOUNT_FACTOR = 0.99f;
constexpr float EPSILON = 0.1f;
constexpr int NUM_EPISODES = 1000;
constexpr int MAX_STEPS = 50;          // Maximum steps per episode

// Pendulum environment
class ControlledPendulum {
private:
    int angle = 30; // Initialize at 30 degrees

public:
    ControlledPendulum() {}

    // Step function: apply action, update angle, and return new state and reward
    std::pair<int, float> step(int action) {
        if (angle < 80) {
            angle += 10; // Only allow increasing the angle
        }

        // Map angle to state space (discrete states: 30, 40, ..., 80)
        int state = (angle - 30) / 10;  // 0, 1, 2, ..., 5 states (30-80 degrees in steps of 10)

        // Reward for moving incrementally within range
        float reward = 1.0f;

        return {state, reward};
    }

    // Reset the pendulum angle for the start of each episode
    void reset() {
        angle = 30;
    }

    // Check if the episode is terminal (angle reaches 80)
    bool is_terminal() {
        return angle >= 80;
    }
};

// Q-learning functions
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
        return 0; // Only one action available (increment)
    }
    return 0; // Always increment (no choice)
}

int main() {
    std::random_device rd;
    std::mt19937 gen(rd());

    // Initialize Q-table with 6 states and 1 action
    auto q_table = torch::zeros({NUM_STATES, NUM_ACTIONS}); // 6 states, 1 action

    for (int episode = 0; episode < NUM_EPISODES; ++episode) {
        ControlledPendulum env;
        env.reset();
        int state = 0;  // Start at angle 30, which corresponds to state 0
        float total_reward = 0;
        int steps = 0;

        float epsilon = std::max(0.01f, EPSILON * std::exp(-0.001f * episode)); // Epsilon decay

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

    // Print the final Q-table after all episodes
    std::cout << "Final Q-table:\n";
    std::cout << "Q-table Shape: [" << NUM_STATES << ", " << NUM_ACTIONS << "]\n";

    for (int i = 0; i < NUM_STATES; ++i) {  // Iterate over all states
        std::cout << "State " << (i * 10) + 30 << ": ";  // Convert state index to actual angle

    for (int j = 0; j < NUM_ACTIONS; ++j) {  // Iterate over all actions
        std::cout << "Q(Action " << j << ") = " << q_table.index({i, j}).item<float>() << ", ";
    }

    std::cout << "\n";
    
    }
    return 0;
}

