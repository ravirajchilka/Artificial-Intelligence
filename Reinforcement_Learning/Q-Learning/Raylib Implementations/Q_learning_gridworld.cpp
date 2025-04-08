#include <torch/torch.h>
#include <raylib.h>
#include <iostream>
#include <cstdint>
#include <random>

constexpr std::int16_t NUM_OF_EPISODES = 500;
constexpr float LEARNING_RATE = 0.15F;
constexpr float DISCOUNT_FACTOR = 0.99F;
constexpr float EPSILON = 0.15F;
constexpr std::int16_t GRID_SIZE = 4; // 4x4 grid

class Grids {
    public:
        Grids(std::int8_t rows_val, std::int8_t cols_val) : rows(rows_val), cols(cols_val) {}

        void draw_grid() const {
            DrawLineEx({0.0f, 0.0f}, {0.0f, 800.0f}, 5.0f, BLACK); // vertical left
            DrawLineEx({0.0f, 0.0f}, {800.0f, 0.0f}, 5.0f, BLACK); // horizontal top
            DrawLineEx({800.0f, 0.0f}, {800.0f, 800.0f}, 5.0f, BLACK); // vertical right
            DrawLineEx({0.0f, 800.0f}, {800.0f, 800.0f}, 5.0f, BLACK); // horizontal bottom

            // draw horizontal lines
            for (std::int8_t i = 1; i < GRID_SIZE; ++i) {
                DrawLineEx({0.0f, static_cast<float>(200 * i)},
                        {800.0f, static_cast<float>(200 * i)}, 5.0f, BLACK);
            }

            // draw vertical lines
            for (std::int8_t i = 1; i < GRID_SIZE; ++i) {
                DrawLineEx({static_cast<float>(i * 200), 0.0f},
                        {static_cast<float>(i * 200), 800.0f}, 5.0f, BLACK);
            }
        }

        void draw_goal_state() const {
            DrawRectangle(600, 600, 200, 200, GREEN);
        }

    private:
        std::int8_t rows;
        std::int8_t cols;
};

class Ball {
    public:
        Ball(Vector2 agent_init_pos, Vector2 goal_pos)
                : ball_pos(agent_init_pos), goal_pos(goal_pos) {}

        void draw_ball() const {
            DrawCircleV(ball_pos, 40, RED);
        }

        void move_ball_forward() {
            if (ball_pos.x < 700) {
                ball_pos.x += 200;
            }
        }

        void move_ball_backward() {
            if (ball_pos.x > 100) {
                ball_pos.x -= 200;
            }
        }

        void move_ball_down() {
            if (ball_pos.y < 700) {
                ball_pos.y += 200;
            }
        }

        void move_ball_up() {
            if (ball_pos.y > 100) {
                ball_pos.y -= 200;
            }
        }

        bool is_terminal_state() const {
            return ball_pos.x == goal_pos.x && ball_pos.y == goal_pos.y;
        }

        void reset_agent_position(Vector2 random_start_pos) {
            ball_pos = random_start_pos;
        }

        std::pair<Vector2, float> get_reward_state_pair(std::int16_t action) {
            switch (action) {
                case 0: move_ball_forward(); break; // Right
                case 1: move_ball_backward(); break; // Left
                case 2: move_ball_up(); break; // Up
                case 3: move_ball_down(); break; // Down
            }

            float reward = is_terminal_state() ? 1.0F : -0.1F;
            return {ball_pos, reward};
        }

    private:
        Vector2 ball_pos;
        Vector2 goal_pos;
};


std::int16_t action_exploration_exploitation(torch::Tensor &q_table, std::mt19937 &gen, float epsilon, Vector2 current_state) {
    static auto real_dis = std::uniform_real_distribution<float>(0.0F, 1.0F);
    static auto int_dis = std::uniform_int_distribution<int>(0, GRID_SIZE - 1);

    // Corrected state calculation
    int state_x = static_cast<int>((current_state.x - 100) / 200);
    int state_y = static_cast<int>((current_state.y - 100) / 200);
    
    if (real_dis(gen) < epsilon) {
        return int_dis(gen); // Exploration: random action
    } else {
        auto state_q_table = q_table[state_x + state_y * GRID_SIZE];
        return state_q_table.argmax().item<int>(); // Exploitation: best action
    }
}

void build_q_table(torch::Tensor &q_table, Vector2 current_state, Vector2 next_state,float reward, std::int16_t action) {
    int current_x = static_cast<int>(current_state.x / 200);
    int current_y = static_cast<int>(current_state.y / 200);
    int next_x = static_cast<int>(next_state.x / 200);
    int next_y = static_cast<int>(next_state.y / 200);

    float max_next_q_value = q_table[next_x + next_y * GRID_SIZE].max().item<float>();
    float target = reward + DISCOUNT_FACTOR * max_next_q_value;

    float current_q_value = q_table[current_x + current_y * GRID_SIZE][action].item<float>();
    q_table[current_x + current_y * GRID_SIZE][action] = current_q_value + LEARNING_RATE * (target - current_q_value);
}

int main() {
    InitWindow(800, 800, "Gridworld");
    SetTargetFPS(60);
    
    Grids grid(GRID_SIZE, GRID_SIZE);

    auto q_table = torch::zeros({GRID_SIZE * GRID_SIZE, GRID_SIZE});
    std::random_device rd;
    std::mt19937 gen(rd());
    auto start_position_dist = std::uniform_int_distribution<int>(0, GRID_SIZE - 1);

    for (std::int16_t episode = 1; episode <= NUM_OF_EPISODES; ++episode) {
        int start_x = start_position_dist(gen);
        int start_y = start_position_dist(gen);

        while (start_x == GRID_SIZE - 1 && start_y == GRID_SIZE - 1) { // Avoid starting at the goal state
            start_x = start_position_dist(gen);
            start_y = start_position_dist(gen);
        }

        Vector2 start_pos = {start_x * 200.f + 100.f, start_y * 200.f + 100.f}; // Ensure valid start position
        Ball ball(start_pos, {700.f, 700.f});
        
        auto current_state = start_pos;
        float total_reward = 0.0f;

        while(!WindowShouldClose() && !ball.is_terminal_state()) {
            std::int16_t action = action_exploration_exploitation(q_table,gen,std::max(static_cast<float>(EPSILON * std::exp(-episode / NUM_OF_EPISODES)), EPSILON),current_state);
            auto [next_state, reward] = ball.get_reward_state_pair(action);
            total_reward += reward;

            build_q_table(q_table, current_state, next_state, reward, action);
            current_state = next_state;

            BeginDrawing();
            ClearBackground(LIGHTGRAY);
            grid.draw_grid();
            grid.draw_goal_state();
            ball.draw_ball();
            EndDrawing();
        }

        // Print the total reward when goal state is reached or episode ends
        if (ball.is_terminal_state()) {
            std::cout << "Episode " << episode << " reached the goal! Total Reward: " << total_reward << '\n';
        }

        if (episode % 100 == 0) {
            std::cout << "Episode: " << episode << ", Total Reward: " << total_reward << '\n';
        }
    }

    std::cout << "Final Q-table:\n";
    for (int i = 0; i < GRID_SIZE * GRID_SIZE; ++i) {
        std::cout << "State " << i << ": ";
        for (int j = 0; j < GRID_SIZE; ++j) {
            std::cout << q_table.index({i, j}).item<float>() << ' ';
        }
        std::cout << '\n';
    }

    CloseWindow();
}
