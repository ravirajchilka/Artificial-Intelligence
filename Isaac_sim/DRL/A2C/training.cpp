#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <torch/torch.h>
#include <random>
#include <cmath>
#include <string>

constexpr int NUM_STATES = 37;
constexpr int NUM_ACTIONS = 2;
constexpr float STEP_DEG = 10.0f;
constexpr float MAX_ANGLE = 360.0f;
constexpr float VALID_ANGLE_MAX = 180.0f;

constexpr float GAMMA = 0.99f;
constexpr float LR = 0.001f;
constexpr int NUM_EPISODES = 500;
constexpr int MAX_STEPS_PER_EPISODE = 50;

constexpr float STEP_PENALTY = -0.03f;
constexpr float ENTROPY_COEF = 0.01f;

struct ActorCriticNet : torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, fc_policy{nullptr}, fc_value{nullptr};

    ActorCriticNet(int state_dim, int action_dim) {
        fc1 = register_module("fc1", torch::nn::Linear(state_dim, 128));
        fc_policy = register_module("fc_policy", torch::nn::Linear(128, action_dim));
        fc_value = register_module("fc_value", torch::nn::Linear(128, 1));
    }

    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x));
        auto policy_logits = fc_policy->forward(x);
        auto value = fc_value->forward(x);
        return {policy_logits, value};
    }
};

class A2CDiskNode : public rclcpp::Node {
private:
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_;

    std::shared_ptr<ActorCriticNet> model_;
    torch::optim::Adam optimizer_;

    int current_state_;
    float last_angle_;
    int episode_;
    int step_;
    std::mt19937 gen_;

public:
    A2CDiskNode() : Node("a2c_disk_node"),
                    model_(std::make_shared<ActorCriticNet>(2, NUM_ACTIONS)),
                    optimizer_(model_->parameters(), torch::optim::AdamOptions(LR)),
                    gen_(std::random_device{}()) {
        publisher_ = create_publisher<std_msgs::msg::String>("topic", 10);

        episode_ = 0;
        step_ = 0;
        last_angle_ = 0.0;

        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(2),
            std::bind(&A2CDiskNode::timer_callback, this));

        start_new_episode();
    }

private:
    void start_new_episode() {
        std::uniform_int_distribution<int> dist(0, NUM_STATES - 1);
        current_state_ = dist(gen_);
        last_angle_ = state_to_angle(current_state_);
        step_ = 0;
        episode_++;
        RCLCPP_INFO(get_logger(), "Episode %d started: initial angle=%.1f", episode_, last_angle_);
    }

    void timer_callback() {
        if (episode_ > NUM_EPISODES) {
            RCLCPP_INFO(get_logger(), "Training complete. Final network outputs:");
            print_nn_table();
            rclcpp::shutdown();
            return;
        }

        torch::Tensor state_tensor = angle_to_input(last_angle_);
        auto [logits, value] = model_->forward(state_tensor);

        torch::Tensor action_prob = torch::softmax(logits, -1);
        std::discrete_distribution<int> action_dist(action_prob.data_ptr<float>(),
                                                    action_prob.data_ptr<float>() + NUM_ACTIONS);
        int action = action_dist(gen_);

        float new_angle = last_angle_ + (action == 0 ? STEP_DEG : -STEP_DEG);
        new_angle = clamp_angle(new_angle, 0.0f, MAX_ANGLE);

        float reward = compute_reward(new_angle) + STEP_PENALTY;

        // Next state value
        torch::Tensor next_state_tensor = angle_to_input(new_angle);
        auto [_, next_value] = model_->forward(next_state_tensor);

        torch::Tensor td_target = torch::tensor({reward}) + GAMMA * next_value.detach();
        torch::Tensor advantage = td_target - value;

        // Actor loss with entropy regularization
        torch::Tensor log_prob = torch::log_softmax(logits, -1)[0][action];
        torch::Tensor entropy = -(torch::softmax(logits, -1) * torch::log_softmax(logits, -1)).sum();
        torch::Tensor actor_loss = -log_prob * advantage.detach() - ENTROPY_COEF * entropy;

        // Critic loss
        torch::Tensor critic_loss = advantage.pow(2);

        torch::Tensor loss = actor_loss + critic_loss;

        optimizer_.zero_grad();
        loss.backward();
        optimizer_.step();

        std_msgs::msg::String msg;
        msg.data = std::to_string(new_angle);
        publisher_->publish(msg);

        last_angle_ = new_angle;
        current_state_ = angle_to_state(new_angle);
        step_++;

        if (step_ >= MAX_STEPS_PER_EPISODE)
            start_new_episode();
    }

    float compute_reward(float angle) {
        if (angle > VALID_ANGLE_MAX) return -10.0f; // scaled down penalty

        float mid = VALID_ANGLE_MAX / 2.0f;
        float dist = std::abs(angle - mid) / (VALID_ANGLE_MAX / 2.0f);
        float reward = 3.0f * (1.0f - dist * dist);

        if (angle > 150.0f)
            reward -= (angle - 150.0f) / 30.0f * 2.0f; // slightly less harsh

        return std::max(reward, -10.0f);
    }

    torch::Tensor angle_to_input(float angle) {
        float rad = angle / VALID_ANGLE_MAX * M_PI; // normalize to 0-pi
        return torch::tensor({std::sin(rad), std::cos(rad)}).unsqueeze(0);
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

    void print_nn_table() {
        for (int i = 0; i < NUM_STATES; ++i) {
            float angle = state_to_angle(i);
            torch::Tensor state_tensor = angle_to_input(angle);
            auto [logits, value] = model_->forward(state_tensor);
            torch::Tensor probs = torch::softmax(logits, -1);

            std::string row = "Angle " + std::to_string(angle) + " -> Action Probabilities: ";
            for (int j = 0; j < NUM_ACTIONS; ++j)
                row += std::to_string(probs[0][j].item<float>()) + " ";
            row += "| Value: " + std::to_string(value.item<float>());
            RCLCPP_INFO(get_logger(), "%s", row.c_str());
        }
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<A2CDiskNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}

