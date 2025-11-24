#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <vector>
#include <cmath>
#include <sstream>

class A2CTablePublisher : public rclcpp::Node
{
public:
    A2CTablePublisher() : Node("a2c_table_publisher"), current_state_(0)
    {
        pub_ = this->create_publisher<std_msgs::msg::String>("topic", 10);

        // NN table: each row = (prob_action0, prob_action1)
        nn_table_ = {
            {0.992366, 0.007634}, {0.999999, 1e-06}, {0.999999, 1e-06}, {0.999999, 1e-06},
            {0.999990, 1e-05}, {0.999757, 0.000243}, {0.998446, 0.001554}, {0.998038, 0.001962},
            {0.993511, 0.006489}, {0.947865, 0.052135}, {0.476016, 0.523984}, {0.020494, 0.979506},
            {0.000249, 0.999751}, {2e-06, 0.999998}, {0, 1}, {0, 1},
            {0, 1}, {0, 1}, {0, 1}, {0, 1},
            {0, 1}, {0, 1}, {0, 1}, {0, 1},
            {0, 1}, {0, 1}, {0, 1}, {0, 1},
            {0, 1}, {0, 1}, {0, 1}, {0, 1},
            {0, 1}, {0.000233, 0.999767}, {0.992367, 0.007633}, {0.992366, 0.007634},
            {0.999999, 1e-06}, {0.999999, 1e-06}, {0.999999, 1e-06}
        };

        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(100),
            std::bind(&A2CTablePublisher::publish_angle, this)
        );
    }

private:
    double angle_from_probs(const std::vector<double>& probs, size_t state_idx)
    {
        // Use probability of action1 to control fine movement
        double p = probs[1];

        // Base angle increments per state
        double base_angle = static_cast<double>(state_idx) * 5.0; // 0,5,10,...
        double angle = base_angle + p * 50.0;  // scale probability to effect

        // Clamp
        if (angle > 180.0) angle = 180.0;
        if (angle < 0.0) angle = 0.0;

        return angle;
    }

    void publish_angle()
    {
        const auto &row = nn_table_[current_state_];
        size_t chosen_action = (row[1] > row[0]) ? 1 : 0;

        int angle_int = static_cast<int>(std::floor(angle_from_probs(row, current_state_)));

        std_msgs::msg::String msg;
        msg.data = std::to_string(angle_int);
        pub_->publish(msg);

        RCLCPP_INFO_STREAM(this->get_logger(),
            "State=" << current_state_
            << ", Chosen action=" << chosen_action
            << ", Probabilities=(" << row[0] << ", " << row[1] << ")"
            << ", Publishing angle=" << angle_int);

        current_state_ = (current_state_ + 1) % nn_table_.size();
    }

    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pub_;
    rclcpp::TimerBase::SharedPtr timer_;
    std::vector<std::vector<double>> nn_table_;
    size_t current_state_;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<A2CTablePublisher>());
    rclcpp::shutdown();
    return 0;
}

