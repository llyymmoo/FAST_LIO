#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CompressedImage.h>
#include <Eigen/Eigen>

#include <iostream>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <string>

#define TIME_SCALE 1000000000

struct Pose {
    std::string timestamp;
    Eigen::Quaterniond qwl;
    Eigen::Vector3d twl;
};

Eigen::Matrix3d Rlc;
Eigen::Vector3d tlc;

void generate_image_pose_list(std::vector<Pose>& ref_poses, std::vector<std::string> image_timestamps, std::string list_save_path)
{
    std::ofstream file;
    file.open(list_save_path, std::ios::app);

    int idx = 0;
    for (auto& timestamp : image_timestamps) {
        while (idx < ref_poses.size() && ref_poses[idx].timestamp < timestamp)
            ++idx;
        
        if (idx == 0) {
            std::cout << "image of timestamp " << timestamp << " is ignored" << std::endl;
            continue;
        }

        if (idx >= ref_pose.size()) {
            std::cout << "image after timestamp " << timestamp << " are ignored" << std::endl;
            break;
        }

        double t1 = stod(ref_poses[idx-1].timestamp) / TIME_SCALE;
        double t2 = stod(ref_poses[idx].timestamp) / TIME_SCALE;
        double t = stod(timestamp) / TIME_SCALE;

        double a1 = (t - t1) / (t2 - t1);
        double a2 = 1 - a1;

        Eigen::Vector4d q1, q2;
        q1(0) = ref_poses[idx-1].qwl.x(); q1(1) = ref_poses[idx-1].qwl.y(); q1(2) = ref_poses[idx-1].qwl.z(); q1(3) = ref_poses[idx-1].qwl.w();
        q2(0) = ref_poses[idx].qwl.x(); q2(1) = ref_poses[idx].qwl.y(); q2(2) = ref_poses[idx].qwl.z(); q2(3) = ref_poses[idx].qwl.w();

        Eigen::Vector4d q = a1*q1 + a2*q2;
        q = q / q.norm();
        Eigen::Vector3d twl = a1*ref_poses[idx-1].twl + a2*ref_poses[idx].twl;
        Eigen::Quaterniond qwl(q);

        Eigen::Matrix3d Rwl = qwl.toRotationMatrix();
        Eigen::Matrix3d Rwc = Rwl * Rlc;
        Eigen::Vector3d twc = Rwl * tlc + twl;
        Eigen::Quaterniond qwc(Rwc);

        file << timestamp
             << qwc.x() << qwc.y() << qwc.z() << qwc.w()
             << twc.x() << twc.y() << twc.z() << std::endl;
    }

    file.close();

    return;
}


int main(int argc, char **argv) {

    // 0. parameters and pathes define
    std::string ros_bag_file_path = "";
    std::string front_image_topic = "";
    std::string back_image_topic = "";
    std::string front_image_save_folder = "";
    std::string back_image_save_folder = "";
    std::string fast_lio_pose_path = "";
    std::string front_image_pose_save_path = "";
    std::string back_image_pose_save_path = "";

    double K[4] = {1, 1, 1, 1};
    double D[5] = {1, 1, 1, 1, 1};
    double time_offset = 0;
    int height = 0;
    int width = 0;

    Rlc;
    tlc;
    
    // 1. extract front and back images and do the undistortion
    rosbag::Bag bag;
    bag.open(ros_bag_file_path, rosbag::bagmode::Read);
    std::vector<std::string> topics;
    topics.push_back(front_image_topic);
    topics.push_back(back_image_topic);
    rosbag::View view(bag, rosbag::TopicQuery(topics));

    cv::Mat camera_matrix = cv::Mat::eye(3, 3, CV_64F);
    camera_matrix.at<double>(0, 0) = K[0];
    camera_matrix.at<double>(0, 2) = K[1];
    camera_matrix.at<double>(1, 1) = K[2];
    camera_matrix.at<double>(1, 2) = K[3];
    cv::Mat distortion_coef = cv::Mat::zeros(4, 1, CV_64F);
    distortion_coef.at<double>(0, 0) = D[0];
    distortion_coef.at<double>(1, 0) = D[1];
    distortion_coef.at<double>(2, 0) = D[2];
    distortion_coef.at<double>(3, 0) = D[3];
    cv::Size imageSize = cv::Size(width, height);
    cv::Mat map1, map2;
    cv::fisheye::initUndistortRectifyMap(camera_matrix, distortion_coef, cv::Matx33d::eye(), camera_matrix, imageSize, CV_16SC2, map1, map2);

    std::vector<std::string> front_image_timestamps, back_image_timestamps;

    for(rosbag::MessageInstance const m : view) {
        std::string topic = m.getTopic();

        sensor_msgs::CompressedImage::ConstPtr img = m.instantiate<sensor_msgs::CompressedImage>();
        cv::Mat image = cv::imdecode(cv::Mat(img->data), cv::IMREAD_COLOR);
        cv::remap(image, image, map1, map2, cv::INTER_LINEAR);

        std::string save_path;
        if (topic == front_image_topic) {
            save_path = front_image_save_folder + "/" + std::to_string(img->header.stamp.toNSec()) + ".png";
            front_image_timestamps.push_back(std::to_string(img->header.stamp.toNSec()));
        }
        else if (topic == back_image_topic) {
            save_path = back_image_save_folder + "/" + std::to_string(img->header.stamp.toNSec()) + ".png";
            back_image_timestamps.push_back(std::to_string(img->header.stamp.toNSec()));
        }
        else {
            std::cout << "Error in topics" << std::endl;
            exit(0);
        }

        cv::imwrite(path, image);
    }

    bag.close();


    // 2. read the trajectory file of fast-lio and interpolate the pose of images
    std::sort(front_image_timestamps.begin(), front_image_timestamps.end(), [](string& a, string& b) {
        return a < b;
    });
    std::sort(back_image_timestamps.begin(), back_image_timestamps.end(), [](string& a, string& b) {
        return a < b;
    });

    std::vector<Pose> lidar_poses;
    std::ifstream lidar_file;
    lidar_file.open(fast_lio_pose_path);
    std::string lidar_line;
    while (getline(lidar_file, lidar_line)) {
        std::stringstream ss(lidar_line);

        Pose pose;
        ss >> pose.timestamp
           >> pose.qwl.x() >> pose.qwl.y() >> pose.qwl.z() >> pose.qwl.w()
           >> pose.twl.x() >> pose.twl.y() >> pose.twl.z();

        lidar_poses.push_back(pose);
    }

    generate_image_pose_list(lidar_poses, front_image_timestamps, front_image_pose_save_path);
    generate_image_pose_list(lidar_poses, back_image_timestamps, back_image_pose_save_path);

    return 0;
}