#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <torch/torch.h>
#include <random>
#include <string>
#include <cmath>

constexpr int NUM_STATES = 37;       // 0 to 360 degrees in 10 deg steps
constexpr int NUM_ACTIONS = 2;       // 0=+10 deg, 1=-10 deg
constexpr float STEP_DEG = 10.0f;
constexpr float MAX_ANGLE = 360.0f;
constexpr float VALID_ANGLE_MAX = 180.0f;

constexpr float ALPHA = 0.12f;
constexpr float GAMMA = 0.99f;
constexpr float EPSILON_INIT = 0.4f;
constexpr float EPSILON_MIN = 0.01f;
constexpr float EPSILON_DECAY = 0.0003f;
constexpr int NUM_EPISODES = 500;
constexpr int MAX_STEPS_PER_EPISODE = 50;

constexpr float GOAL_MIN = 0.0f;
constexpr float GOAL_MAX = 180.0f;

constexpr float STEP_PENALTY = -0.03f;
constexpr int STUCK_STEP_THRESHOLD = 6;

class QLearningDiskNode : public rclcpp::Node {
public:
    QLearningDiskNode() : Node("q_learning_disk_node"), gen_(rd_()) {
        publisher_ = create_publisher<std_msgs::msg::String>("topic", 10);
        q_table_ = torch::zeros({NUM_STATES, NUM_ACTIONS});
        epsilon_ = EPSILON_INIT;
        episode_ = 0;
        step_ = 0;
        stuck_steps_ = 0;
        last_angle_ = 0.0;

        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(5),
            std::bind(&QLearningDiskNode::timer_callback, this));

        start_new_episode();
    }

private:
    void start_new_episode() {
        std::uniform_int_distribution<int> dist(0, NUM_STATES - 1);
        current_state_ = dist(gen_);
        last_angle_ = state_to_angle(current_state_);
        step_ = 0;
        stuck_steps_ = 0;
        episode_++;
        epsilon_ = std::max(epsilon_, EPSILON_INIT * 0.5f);
        RCLCPP_INFO(get_logger(), "Episode %d started: initial angle=%.1f, epsilon=%.3f",
                    episode_, last_angle_, epsilon_);
    }

    void timer_callback() {
        if (episode_ > NUM_EPISODES) {
            RCLCPP_INFO(get_logger(), "Training complete. Final Q-table:");
            print_q_table();
            rclcpp::shutdown();
            return;
        }

        int action = choose_action(current_state_);
        float new_angle = last_angle_ + (action == 0 ? STEP_DEG : -STEP_DEG);

        // Clamp angle within 0-360 for state index
        new_angle = clamp_angle(new_angle, 0.0f, MAX_ANGLE);
        int next_state = angle_to_state(new_angle);

        // If next angle > 180, give strong punishment and reset next_state to current_state
        float reward;
        if (new_angle > VALID_ANGLE_MAX) {
            reward = -50.0f;  // Strong negative reward
            next_state = current_state_;  // Do not allow agent to move beyond 180
        } else {
            reward = compute_reward(new_angle);
        }
        reward += STEP_PENALTY;

        update_q_table(reward, action, next_state, current_state_);

        std_msgs::msg::String msg;
        msg.data = std::to_string(new_angle);
        publisher_->publish(msg);

        RCLCPP_INFO(get_logger(), "angle=%.1f, reward=%.2f, step=%d", new_angle, reward, step_);

        // Detect if stuck near 180 degrees (within one step)
        if (std::abs(last_angle_ - VALID_ANGLE_MAX) <= STEP_DEG &&
            std::abs(new_angle - VALID_ANGLE_MAX) <= STEP_DEG)
            stuck_steps_++;
        else
            stuck_steps_ = 0;

        if (stuck_steps_ >= STUCK_STEP_THRESHOLD) {
            epsilon_ = std::min(0.6f, epsilon_ + 0.1f);
            RCLCPP_WARN(get_logger(), "Agent stuck near 180, boosting epsilon to %.3f", epsilon_);
            stuck_steps_ = 0;
        }

        current_state_ = next_state;
        last_angle_ = new_angle;
        step_++;

        epsilon_ = std::max(EPSILON_MIN, epsilon_ * std::exp(-EPSILON_DECAY));

        if (step_ >= MAX_STEPS_PER_EPISODE) {
            start_new_episode();
        }
    }

    int choose_action(int state) {
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        if (dist(gen_) < epsilon_) {
            std::uniform_int_distribution<int> adist(0, NUM_ACTIONS - 1);
            return adist(gen_);
        }
        // Mask invalid actions (prevent moving >180)
        torch::Tensor q_values = q_table_.index({state});
        if (state_to_angle(state) >= VALID_ANGLE_MAX && q_values[0].item<float>() < q_values[1].item<float>())
            return 1;  // Force -10 deg if at or above 180
        if (state_to_angle(state) >= VALID_ANGLE_MAX && q_values[1].item<float>() < q_values[0].item<float>())
            return 0;  // Force +10 deg if at or above 180
        return q_values.argmax().item<int>();
    }

    void update_q_table(float reward, int action, int next_state, int state) {
        // Only consider max_next if next_state is valid
        float max_next = 0.0f;
        if (state_to_angle(next_state) <= VALID_ANGLE_MAX)
            max_next = q_table_.index({next_state}).max().item<float>();

        float current_q = q_table_.index({state, action}).item<float>();
        float target = reward + GAMMA * max_next;
        float td_error = target - current_q;
        float new_q = current_q + ALPHA * td_error;
        q_table_.index_put_({state, action}, new_q);
    }

    // float compute_reward(float angle) {
    //     float mid = (GOAL_MIN + GOAL_MAX) / 2.0f;
    //     float dist = std::abs(angle - mid) / (VALID_ANGLE_MAX / 2.0f);
    //     float reward = 3.0f * (1.0f - dist * dist);
    //     return std::max(reward, 0.0f);
    // }

    float compute_reward(float angle) {
        if (angle > VALID_ANGLE_MAX) {
            return -50.0f;  // strong negative
        }
        // Smooth reward: maximum at 90 degrees (mid)
        float mid = (GOAL_MIN + GOAL_MAX) / 2.0f;  // 90 deg
        float dist = std::abs(angle - mid) / (VALID_ANGLE_MAX / 2.0f); // 0..1
        float reward = 3.0f * (1.0f - dist*dist);  // quadratic drop
        // gradually penalize near 180
        if (angle > 150.0f)
            reward -= (angle - 150.0f)/30.0f * 3.0f; // -0..3
        return std::max(reward, -50.0f); 
    }


    float clamp_angle(float angle, float min_val, float max_val) {
        return std::min(std::max(angle, min_val), max_val);
    }

    int angle_to_state(float angle) {
        int s = static_cast<int>(std::round(angle / STEP_DEG));
        return std::min(std::max(s, 0), NUM_STATES - 1);
    }

    float state_to_angle(int s) {
        return s * STEP_DEG;
    }

    void print_q_table() {
        for (int i = 0; i < NUM_STATES; ++i) {
            float angle = state_to_angle(i);
            std::string row = "State " + std::to_string(i) + " (Angle " + std::to_string(angle) + "): ";
            for (int j = 0; j < NUM_ACTIONS; ++j) {
                row += std::to_string(q_table_.index({i, j}).item<float>()) + " ";
            }
            RCLCPP_INFO(get_logger(), "%s", row.c_str());
        }
    }

    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_;

    torch::Tensor q_table_;
    int current_state_;
    float last_angle_;
    float epsilon_;
    int episode_;
    int step_;
    int stuck_steps_;

    std::random_device rd_;
    std::mt19937 gen_;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<QLearningDiskNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}


