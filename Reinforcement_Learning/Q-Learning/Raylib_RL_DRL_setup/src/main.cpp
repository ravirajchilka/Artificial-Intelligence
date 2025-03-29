#include <torch/torch.h>
#include <iostream>
#include <random>
#include <algorithm>
#include "raylib.h"

constexpr int GRID_SIZE = 4;
constexpr int NUM_ACTIONS = 4;
constexpr int NUM_EPISODES = 1000;
constexpr float LEARNING_RATE = 0.1;
constexpr float DISCOUNT_FACTOR = 0.99;
constexpr float EPSILON = 0.1;
constexpr int SCREEN_WIDTH = 800;
constexpr int SCREEN_HEIGHT = 800;

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

    void reset() { agent_pos = 0; }
    bool is_terminal() { return agent_pos == goal_pos; }
    int get_agent_pos() const { return agent_pos; }
    int get_goal_pos() const { return goal_pos; }
};

void draw_grid(const GridWorld& env) {
    float cell_size = SCREEN_WIDTH / GRID_SIZE;
    for (int i = 0; i < GRID_SIZE; ++i) {
        for (int j = 0; j < GRID_SIZE; ++j) {
            Vector2 cell_pos = {i * cell_size, j * cell_size};
            Color color = (i * GRID_SIZE + j == env.get_goal_pos()) ? GREEN : LIGHTGRAY;
            DrawRectangleV(cell_pos, {cell_size, cell_size}, color);
            DrawRectangleLinesEx({cell_pos.x, cell_pos.y, cell_size, cell_size}, 2, BLACK);
        }
    }
    int agent_pos = env.get_agent_pos();
    int agent_row = agent_pos / GRID_SIZE;
    int agent_col = agent_pos % GRID_SIZE;
    Vector2 agent_pos_in_grid = {agent_col * cell_size + cell_size / 2, agent_row * cell_size + cell_size / 2};
    DrawCircleV(agent_pos_in_grid, cell_size / 4, BLUE);
}

int main() {
    InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "GridWorld Q-learning");
    SetTargetFPS(60);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    auto q_table = torch::zeros({GRID_SIZE * GRID_SIZE, NUM_ACTIONS});
    GridWorld env(0, GRID_SIZE * GRID_SIZE - 1);

    for (int episode = 0; episode < NUM_EPISODES; ++episode) {
        env.reset();
        int state = 0;
        float total_reward = 0;

        while (!env.is_terminal()) {
            int action;
            if (dis(gen) < EPSILON) {
                action = std::uniform_int_distribution<>(0, NUM_ACTIONS - 1)(gen);
            } else {
                action = q_table[state].argmax().item<int>();
            }

            auto [next_state, reward] = env.step(action);
            total_reward += reward;

            float max_next_q = q_table[next_state].max().item<float>();
            float target = reward + DISCOUNT_FACTOR * max_next_q;
            float current_q = q_table[state][action].item<float>();
            q_table[state][action] = current_q + LEARNING_RATE * (target - current_q);

            state = next_state;

            BeginDrawing();
            ClearBackground(RAYWHITE);
            draw_grid(env);
            EndDrawing();

            WaitTime(0.1);  // Slow down movement
        }

       // if (episode % 100 == 0) {
            std::cout << "Episode --> " << episode << ", Total Reward: " << total_reward << std::endl;
       // }
    }

    CloseWindow();
    return 0;
}
