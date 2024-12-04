#include "ros/ros.h"
#include "std_msgs/Float32MultiArray.h"
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>
#include <vector>

class GelloReplayer {
public:
    GelloReplayer(const std::string& csv_path) : nh_(), loop_rate_(30) {
        // Initialize publisher
        gello_pub_ = nh_.advertise<std_msgs::Float32MultiArray>("gello_replay", 10);

        // Open the CSV file
        file_.open(csv_path);
        if (!file_.is_open()) {
            ROS_ERROR("Failed to open CSV file for reading: %s", csv_path.c_str());
            ros::shutdown();
        } else {
            ROS_INFO("Opened Replay file: %s", csv_path.c_str());
        }
        ROS_INFO("Ready to Replay Gello Data");
    }

    ~GelloReplayer() {
        if (file_.is_open()) {
            file_.close();
        }
    }

    void spin() {
        std::string line;

        while (ros::ok() && std::getline(file_, line)) {
            float gello_data;
            if (parseLine(line, gello_data)) {
                publishGelloData(gello_data);
            } else {
                ROS_WARN("Failed to parse line: %s", line.c_str());
            }

            ros::spinOnce();
            loop_rate_.sleep();
        }

        ROS_INFO("Finished replaying log file.");
        ros::shutdown();
    }

private:
    ros::NodeHandle nh_;
    ros::Publisher gello_pub_;
    ros::Rate loop_rate_;
    std::ifstream file_;

    bool parseLine(const std::string& line, float& gello_data) {
        std::stringstream ss(line);
        std::string value;

        // Skip the first column (time)
        std::getline(ss, value, ',');

        // Get the gello_data
        if (std::getline(ss, value, ',')) {
            try {
                gello_data = std::stof(value);
                return true;
            } catch (const std::invalid_argument& e) {
                ROS_ERROR("Invalid float value in CSV: %s", value.c_str());
                return false;
            }
        } else {
            ROS_WARN("Failed to parse gello_data from line: %s", line.c_str());
            return false;
        }
    }

    void publishGelloData(float gello_data) {
        std_msgs::Float32MultiArray gello_msg;
        gello_msg.data.push_back(gello_data);
        gello_pub_.publish(gello_msg);
        ROS_INFO("Published Gello data: [%f]", gello_data);
    }
};

// Function to extract directory from __FILE__
std::string getCodeDirectory() {
    std::string path = __FILE__;
    std::size_t pos = path.find_last_of("/\\");
    return (pos == std::string::npos) ? "" : path.substr(0, pos);
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "gello_replayer");

    // CSV file path
    std::string codeDir = getCodeDirectory();
    std::cout << "Code's directory: " << codeDir << std::endl;
    std::string csv_path = codeDir + "/gello_log.csv";

    GelloReplayer replayer(csv_path);
    replayer.spin();
    ros::shutdown();

    return 0;
}
