#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <vector>
#include <cmath>
#include <sstream>

class QTablePublisher : public rclcpp::Node
{
public:
    QTablePublisher() : Node("q_table_publisher"), current_state_(0)
    {
        pub_ = this->create_publisher<std_msgs::msg::String>("topic", 10);

        q_table_ = {
            {73.445335, 2.536616}, {138.213104, 16.840034}, {196.238098, 16.213596},
            {223.991287, 102.773758}, {234.168091, 181.483673}, {235.325089, 232.611572},
            {232.422058, 235.204651}, {164.129623, 233.795319}, {43.138660, 217.594284},
            {4.315781, 148.335358}, {64.950317, 1.396924}, {136.378036, 2.377108},
            {161.559677, 69.031586}, {168.225891, 135.722839}, {169.322021, 167.285416},
            {165.355698, 169.535645}, {135.898895, 168.446823}, {52.472244, 161.702301},
            {-5.456388, 143.863754}, {-43.559273, 30.013643}
        };

        compute_q_min_max();

        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(100),
            std::bind(&QTablePublisher::publish_angle, this)
        );
    }

private:
    void compute_q_min_max()
    {
        min_q_ = 1e9;
        max_q_ = -1e9;
        for (const auto &row : q_table_) {
            for (double v : row) {
                min_q_ = std::min(min_q_, v);
                max_q_ = std::max(max_q_, v);
            }
        }
        if (min_q_ == max_q_) max_q_ = min_q_ + 1.0;
    }

 double map_to_servo_range(double q_value) const
{
    // Define a practical Q range for servo
    double q_min = 0.0;
    double q_max = 180.0; // or pick a smaller "useful max" like 170

    // Clamp
    if (q_value < q_min) q_value = q_min;
    if (q_value > q_max) q_value = q_max;

    // Linear map
    double angle = (q_value - q_min) / (q_max - q_min) * 180.0;

    // Floor and clamp again just in case
    int angle_int = static_cast<int>(std::floor(angle));
    if (angle_int < 0) angle_int = 0;
    if (angle_int > 180) angle_int = 180;

    return static_cast<double>(angle_int);
}


    void publish_angle()
    {
        const auto &row = q_table_[current_state_];
        size_t max_action = (row[1] > row[0]) ? 1 : 0;
        double best_q = row[max_action];

        // map and floor
        int angle_int = static_cast<int>(std::floor(map_to_servo_range(best_q)));

        // send as STRING
        std_msgs::msg::String msg;
        msg.data = std::to_string(angle_int);
        pub_->publish(msg);

        RCLCPP_INFO_STREAM(this->get_logger(),
            "State=" << current_state_
            << ", Action=" << max_action
            << ", Q=" << best_q
            << ", Publishing angle=" << angle_int);

        current_state_ = (current_state_ + 1) % q_table_.size();
    }

    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pub_;
    rclcpp::TimerBase::SharedPtr timer_;
    std::vector<std::vector<double>> q_table_;
    size_t current_state_;
    double min_q_, max_q_;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<QTablePublisher>());
    rclcpp::shutdown();
    return 0;
}


