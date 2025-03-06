#include <iostream>
#include <random>
#include <torch/torch.h>

constexpr int NUM_ACTIONS = 2;         // Two actions: increase angle or decrease angle
constexpr int NUM_STATES = 18;         // 18 possible states (angle 0, 10, 20, ..., 180)
constexpr float LEARNING_RATE = 0.1f;
constexpr float DISCOUNT_FACTOR = 0.99f;
constexpr float EPSILON = 0.1f;
constexpr int NUM_EPISODES = 1000;
constexpr int MAX_STEPS = 50;          // Maximum steps per episode

// Pendulum environment
class ControlledPendulum {
private:
    int angle = 0; // Initialize angle at 0

public:
    ControlledPendulum() {}

    // Step function: apply action, update angle, and return new state and reward
    std::pair<int, float> step(int action) {
        switch (action) {
            case 0: angle += 10; break; // Increase angle
            case 1: angle -= 10; break; // Decrease angle
        }

        if (angle < 0) angle = 0; // Prevent going below 0
        if (angle > 180) angle = 180; // Prevent going beyond 180

        // Map angle to state space (discrete states)
        int state = angle / 10;  // 0, 1, 2, ..., 18 states (0-180 degrees in steps of 10)

        // Define reward: higher when within the desired angle range (30 to 80 degrees)
        float reward = (angle >= 30 && angle <= 80) ? 1.0f : 0.1f;

        return {state, reward};
    }

    // Reset the pendulum angle for the start of each episode
    void reset() {
        angle = 0;
    }

    // Check if the episode is terminal (angle reaches 180)
    bool is_terminal() {
        return angle == 120;
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
        return std::uniform_int_distribution<>(0, NUM_ACTIONS - 1)(gen); // Random action (exploration)
    }
    return q_table.index({state}).argmax().item<int>(); // Best action (exploitation)
}

int main() {
    std::random_device rd;
    std::mt19937 gen(rd());

    // Initialize Q-table with 18 states and 2 actions
    auto q_table = torch::zeros({NUM_STATES, NUM_ACTIONS}); // 18 states, 2 actions

    for (int episode = 0; episode < NUM_EPISODES; ++episode) {
        ControlledPendulum env;
        env.reset();
        int state = 0;  // Start at angle 0, which corresponds to state 0
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
    std::cout << "Final Q-table:" << std::endl;
    for (int i = 0; i < NUM_STATES; ++i) {  // For each of the 18 possible states
        std::cout << "State " << i * 10 << ": "; // Display the angle corresponding to the state
        for (int j = 0; j < NUM_ACTIONS; ++j) {
            std::cout << q_table.index({i, j}).item<float>() << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
