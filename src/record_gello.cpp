#include "ros/ros.h"
#include "std_msgs/Float32MultiArray.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <mutex>
#include <signal.h>

class GelloLogger {
public:
    GelloLogger(const std::string& csv_path) : nh_(), loop_rate_(30) {
        // Initialize subscriber
        gello_sub_ = nh_.subscribe("gello_data", 10, &GelloLogger::gelloCallback, this);

        start_time_ = ros::Time::now().toSec();

        // Open CSV file for writing
        csv_file_.open(csv_path);
        if (!csv_file_.is_open()) {
            ROS_ERROR("Failed to open CSV file for writing.");
            ros::shutdown();
        }
        ROS_INFO("File opened. Recording...");

        // Register signal handler for graceful shutdown
        signal(SIGINT, GelloLogger::signalHandler);
    }

    ~GelloLogger() {
        if (csv_file_.is_open()) {
            csv_file_.close();
        }
    }

    void spin() {
        while (ros::ok() && is_running_) {
            if (data_received_) {
                float gello_value = getGelloData();
                double elapsed_time = ros::Time::now().toSec() - start_time_;
                if (csv_file_.is_open()) {

                    csv_file_ << elapsed_time << "," << gello_value << std::endl;
                }

                data_received_ = false;  // Reset the flag
            }

            ros::spinOnce();
            loop_rate_.sleep();
        }
        ROS_INFO("Finished recording.");
        ros::shutdown();
    }

private:
    ros::NodeHandle nh_;
    ros::Subscriber gello_sub_;
    ros::Rate loop_rate_;
    std::ofstream csv_file_;
    static bool is_running_;

    float gello_data_{0.0f};
    bool data_received_ = false;
    static std::mutex data_mutex_;
    double start_time_;

    static void signalHandler(int signum) {
        ROS_INFO("Interrupt signal received. Saving CSV file and exiting...");
        is_running_ = false;
        ros::shutdown();
    }

    void gelloCallback(const std_msgs::Float32MultiArray::ConstPtr& msg) {
        std::lock_guard<std::mutex> lock(data_mutex_);
        if (!msg->data.empty()) {
            gello_data_ = msg->data[0];  // Assuming the data contains one float value
            data_received_ = true;
        }
    }

    float getGelloData() {
        std::lock_guard<std::mutex> lock(data_mutex_);
        return gello_data_;
    }
};

// Initialize static member variables
bool GelloLogger::is_running_ = true;
std::mutex GelloLogger::data_mutex_;

// Function to extract directory from __FILE__
std::string getCodeDirectory() {
    std::string path = __FILE__;
    std::size_t pos = path.find_last_of("/\\");
    return (pos == std::string::npos) ? "." : path.substr(0, pos);
}

int main(int argc, char **argv) {
    // Initialize the ROS node
    ros::init(argc, argv, "gello_recorder");

    // CSV file path
    std::string codeDir = getCodeDirectory();
    std::cout << "Code's directory: " << codeDir << std::endl;
    std::string csv_path = codeDir + "/gello_log.csv";

    GelloLogger logger(csv_path);
    sleep(1); // Wait for 1 second before starting
    logger.spin();
    return 0;
}
