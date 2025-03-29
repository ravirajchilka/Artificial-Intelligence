#include "raylib.h"
#include <vector>
#include <iostream>
#include <cstdlib>
#include <ctime>

constexpr int screen_width = 400;
constexpr int screen_height = 400;

// Grid dimensions
constexpr int grid_rows = 4;
constexpr int grid_cols = 4;
constexpr int cell_size = screen_width / grid_cols;

// Box dimensions
constexpr int box_size = cell_size;  // Box will occupy one cell in the grid

// Box position on the grid
int box_row = 0;
int box_col = 0;

// Actions: Up, Down, Left, Right
enum Action { UP = 0, DOWN, LEFT, RIGHT };

// Q-Learning parameters
constexpr float alpha = 0.1;  // Learning rate
constexpr float gamma = 0.9;  // Discount factor
constexpr float epsilon = 0.1; // Exploration rate

// Q-table: 16 states (0-15) and 4 actions (Up, Down, Left, Right)
std::vector<std::vector<float>> Q_table(16, std::vector<float>(4, 0.0f));

// Reward function (Goal is state 15)
int getReward(int state) {
    if (state == 15) {
        return 1; // Goal reached
    }
    return -1; // No reward otherwise
}

// Convert row-col position to state number (0 to 15)
int toState(int row, int col) {
    return row * grid_cols + col;
}

// Choose action using epsilon-greedy policy
Action chooseAction(int state) {
    if ((rand() % 100) < epsilon * 100) {
        // Exploration: random action
        return static_cast<Action>(rand() % 4);
    }
    // Exploitation: choose best action based on Q-table
    float maxQ = Q_table[state][0];
    Action best_action = UP;
    for (int a = 1; a < 4; ++a) {
        if (Q_table[state][a] > maxQ) {
            maxQ = Q_table[state][a];
            best_action = static_cast<Action>(a);
        }
    }
    return best_action;
}

// Take action and return the next state and reward
void takeAction(int &row, int &col, Action action) {
    switch (action) {
        case UP:    if (row > 0) row--; break;
        case DOWN:  if (row < grid_rows - 1) row++; break;
        case LEFT:  if (col > 0) col--; break;
        case RIGHT: if (col < grid_cols - 1) col++; break;
    }
}

// Update Q-table using Q-learning formula
void updateQTable(int state, Action action, int nextState, int reward) {
    float maxQ_next = *std::max_element(Q_table[nextState].begin(), Q_table[nextState].end());
    Q_table[state][action] = Q_table[state][action] + alpha * (reward + gamma * maxQ_next - Q_table[state][action]);
}

int main() {
    srand(static_cast<unsigned int>(time(0)));

    // Initialize window
    InitWindow(screen_width, screen_height, "Q-Learning in Grid World");
    SetTargetFPS(60);  // Set FPS to 60

    // Training loop (Q-learning)
    for (int episode = 0; episode < 1000; ++episode) {
        box_row = 0;  // Start from (0,0)
        box_col = 0;

        int state = toState(box_row, box_col);
        int totalReward = 0;

        while (state != 15) {  // Until the goal is reached (state 15)
            // Choose an action
            Action action = chooseAction(state);

            // Take the action and observe the new state and reward
            int prev_row = box_row;
            int prev_col = box_col;
            takeAction(box_row, box_col, action);

            int nextState = toState(box_row, box_col);
            int reward = getReward(nextState);

            // Update the Q-table based on the observed reward and next state
            updateQTable(state, action, nextState, reward);

            state = nextState;
            totalReward += reward;

            // Optionally, you can add a small delay here for better visualization
        }
    }

    // Visualization loop
    while (!WindowShouldClose()) {
        // Start drawing
        BeginDrawing();
        ClearBackground(RAYWHITE);

        // Draw grid
        for (int i = 0; i < grid_rows; ++i) {
            for (int j = 0; j < grid_cols; ++j) {
                // Draw grid cells
                DrawRectangle(j * cell_size, i * cell_size, cell_size, cell_size, LIGHTGRAY);
                DrawRectangleLines(j * cell_size, i * cell_size, cell_size, cell_size, DARKGRAY);
            }
        }

        // Draw the box (moving rectangle)
        DrawRectangle(box_col * cell_size, box_row * cell_size, box_size, box_size, BLUE);

        // Update agent position based on the learned policy
        int state = toState(box_row, box_col);
        Action best_action = chooseAction(state);
        takeAction(box_row, box_col, best_action);

        // End drawing
        EndDrawing();
    }

    // De-Initialization
    CloseWindow();

    return 0;
}
