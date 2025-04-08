#include <torch/torch.h>
#include "raylib.h"
#include <cstdint>
#include <iostream>
#include <random>
#include <cmath>

constexpr std::int16_t screen_width = 800;
constexpr std::int16_t screen_height = 600;

constexpr std::int16_t NUM_ACTIONS = 2;      // Two actions: increase or decrease rotation
constexpr std::int16_t NUM_EPISODES = 100;  // Number of training episodes
constexpr std::int16_t NUM_STATES = 91;     // Discrete states (angles 0 to 90)
constexpr float LEARNING_RATE = 0.15F;
constexpr float EPSILON_ = 0.1F;
constexpr float DISCOUNT_FACTOR = 0.99F;

class Disk {
public:
    float rotation_angle = 0.0F;  // Current rotation angle
    float reward = 0.0F;
    Vector2 center = Vector2{screen_width / 2.0f, screen_height / 2.0f}; // Center of the disk

    Disk() = default;

    void draw_disk_with_strip() {
        // Define the rectangle (strip)
        Rectangle strip = Rectangle{center.x - strip_width / 2, center.y - strip_height / 2, strip_width, strip_height};

        // Draw the circle and the rotating strip
        DrawCircle(center.x, center.y, strip_height / 2, RED);
        DrawRectanglePro(strip, Vector2{strip.width / 2, strip.height / 2}, rotation_angle, BLUE);
    }

    std::pair<float, int> get_reward_and_state_pair(std::int16_t action) {
        float prev_rotation_angle = rotation_angle;

        // Update rotation based on action
        if (action == 0) {
            rotation_angle += 1.0F; // Increase angle
        } else if (action == 1) {
            rotation_angle -= 1.0F; // Decrease angle
        }

        // Clamp rotation angle to valid range [0, 90]
        rotation_angle = std::clamp(rotation_angle, 0.0F, 90.0F);

        // Calculate reward: higher reward for getting closer to target (80 degrees)
        if (rotation_angle >= 80) {
            reward = 10.0F; // Terminal reward for reaching target
        } else {
            reward = (rotation_angle - prev_rotation_angle) * 0.1F; // Small incremental reward
        }

        // Convert continuous rotation angle to discrete state
        int state = static_cast<int>(rotation_angle);

        return {reward, state};
    }

    bool is_terminal() const {
        return rotation_angle >= 80; // Terminal state when angle reaches or exceeds 80 degrees
    }

    void reset() {
        rotation_angle = 0.0F; // Reset angle to initial state
    }

private:
    float strip_width = 20.0F;   // Width of the strip
    float strip_height = 100.0F; // Height of the strip
};

// Function to choose an action using epsilon-greedy policy
std::int16_t return_action_with_exploration_exploitation(torch::Tensor &q_table, int state, std::mt19937& gen) {
    std::uniform_real_distribution<> dis(0.0, 1.0);
    if (dis(gen) < EPSILON_) {
        return std::uniform_int_distribution<>(0, NUM_ACTIONS - 1)(gen); // Random action (exploration)
    }
    return q_table.index({state}).argmax().item<int>(); // Best action (exploitation)
}

// Q-learning function to update the Q-table
void update_q_table(torch::Tensor& q_table, int state, int action, float reward, int next_state, bool is_terminal) {
    float max_next_q = is_terminal ? 0 : q_table.index({next_state}).max().item<float>();  
    float target = reward + DISCOUNT_FACTOR * max_next_q;  
    float current_q = q_table.index({state, action}).item<float>();  
    q_table.index_put_({state, action}, current_q + LEARNING_RATE * (target - current_q));
}

int main() {
    std::random_device rd;
    std::mt19937 gen(rd());

    InitWindow(screen_width, screen_height, "Rotating Disk Q-Learning");
    SetTargetFPS(60);

    torch::Tensor q_table = torch::zeros({NUM_STATES, NUM_ACTIONS}); // Initialize Q-table with zeros

    for (int episode = 0; episode < NUM_EPISODES; ++episode) {
        Disk env;
        env.reset();
        
        int state = static_cast<int>(env.rotation_angle); // Start at initial state
        float total_reward = 0;

        while (!WindowShouldClose() && !env.is_terminal()) {
            int action = return_action_with_exploration_exploitation(q_table, state, gen); // Choose an action
            auto [reward, next_state] = env.get_reward_and_state_pair(action);            // Get reward and next state
            
            total_reward += reward;

            update_q_table(q_table, state, action, reward, next_state, env.is_terminal()); // Update Q-table
            
            state = next_state;

            BeginDrawing();
                ClearBackground(RAYWHITE);
                env.draw_disk_with_strip(); // Draw the environment
            EndDrawing();
        }

        std::cout << "Episode " << episode << ", Total Reward: " << total_reward << std::endl;
    }

    // Print the final Q-table after training
    std::cout << "Final Q-table:" << std::endl;
    for (int i = 0; i < NUM_STATES; ++i) { 
        std::cout << "State " << i << ": ";
        for (int j = 0; j < NUM_ACTIONS; ++j) {
            std::cout << q_table.index({i, j}).item<float>() << " ";
        }
        std::cout << std::endl;
    }

    CloseWindow();
    return 0;
}
