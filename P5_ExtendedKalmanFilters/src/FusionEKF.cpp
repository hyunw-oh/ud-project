#include "FusionEKF.h"
#include <iostream>
#include "Eigen/Dense"
#include "tools.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;
using std::vector;

/**
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd::Identity(2,4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
              0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
              0, 0.0009, 0,
              0, 0, 0.09;


  cout<<"Reached till f";

  ekf_.F_ = MatrixXd::Identity(4,4);

  /**
   * TODO: Finish initializing the FusionEKF.
   * TODO: Set the process and measurement noises
   */


}

/**
 * Destructor.
 */
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {
  /**
   * Initialization
   */
  if (!is_initialized_) {
    /**
     * TODO: Initialize the state ekf_.x_ with the first measurement.
     * TODO: Create the covariance matrix.
     * You'll need to convert radar from polar to cartesian coordinates.
     */

    previous_timestamp_ = measurement_pack.timestamp_;

    // first measurement
    std::cout << "EKF: " << std::endl;
    ekf_.x_ = VectorXd(4);
    ekf_.x_ << 1, 1, 1, 1;
    ekf_.P_ = MatrixXd::Identity(4,4);

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      */
      double ro = measurement_pack.raw_measurements_(0);
      double theta= measurement_pack.raw_measurements_(1);
      double ro_theta = measurement_pack.raw_measurements_(2);
      ekf_.x_ << ro * cos(theta)
                ,ro * sin(theta)
                ,ro_theta * cos(theta)
                ,ro_theta * sin(theta);
      //vx = ?
      //vy = ?

    }  else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      /**
      Initialize state.
      */
      std::cout<<"Timestamp2"<< std::endl;
      ekf_.x_<<measurement_pack.raw_measurements_(0)
              ,measurement_pack.raw_measurements_(1)
              ,0.0
              ,0.0;
    }
    ekf_.x_(2) = 5.2;
    ekf_.x_(3) = 1.8/1000.0;

    // done initializing, no need to predict or update
    is_initialized_ = true;

    std::cout<<"out init of ProcessMeasurement"<<std::endl;
    return;
  }

  /**
   * Prediction
   */

  /**
   * TODO: Update the state transition matrix F according to the new elapsed time.
   * Time is measured in seconds.
   * TODO: Update the process noise covariance matrix.
   * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
   */

  double dt = (measurement_pack.timestamp_ - previous_timestamp_ ) / 1000000.0;
  previous_timestamp_ = measurement_pack.timestamp_;

  float noise_ax=9.0;
  float noise_ay=9.0;

  double dt2 = dt * dt;
  double dt3 = dt2 * dt;
  double dt4 = dt2 * dt2;

  dt3 /= 2;
  dt4 /= 4;
  ekf_.Q_= MatrixXd(4,4);
  ekf_.Q_<< dt4 * noise_ax, 0, dt3 * noise_ax, 0,
            0, dt4 * noise_ay, 0, dt3  * noise_ay,
            dt3 * noise_ax, 0, dt2 * noise_ax, 0,
            0, dt3 * noise_ay, 0, dt2 * noise_ay;

  ekf_.F_(0,2)= dt;
  ekf_.F_(1,3)= dt;

  ekf_.Predict();

  /**
   * Update
   */

  /**
   * TODO:
   * - Use the sensor type to perform the update step.
   * - Update the state and covariance matrices.
   */

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR){
    // Radar updates
    ekf_.H_ = tools.CalculateJacobian(ekf_.x_);
    ekf_.R_= R_radar_;
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  } else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
    // Laser updates
    ekf_.H_= H_laser_;
    ekf_.R_ = R_laser_;
    ekf_.Update(measurement_pack.raw_measurements_);
  }

  // print the output
  std::cout << "x_ = " << ekf_.x_ << std::endl;
  std::cout << "P_ = " << ekf_.P_ << std::endl;
}
