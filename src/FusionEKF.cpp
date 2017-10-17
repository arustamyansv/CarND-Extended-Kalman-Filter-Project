#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // Measurement covariance matrix - laser
  R_laser_ = MatrixXd(2, 2);
  R_laser_ << 0.0225, 0,
              0,      0.0225;

  // Measurement matrix - laser
  H_laser_ = MatrixXd(2, 4);
  H_laser_ << 1, 0, 0, 0,
              0, 1, 0, 0;


  // Measurement covariance matrix - radar
  R_radar_ = MatrixXd(3, 3);
  R_radar_ << 0.09, 0,      0,
              0,    0.0009, 0,
              0,    0,      0.09;

  // Measurement matrix - radar
  H_radar_ = MatrixXd(3, 4);
  H_radar_ << 1, 1, 0, 0,
              1, 1, 0, 0,
              1, 1, 1, 1;

  // Process covariance matrix params
  Q_noise_ax_ = 9;
  Q_noise_ay_ = 9;
}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {


  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {

    ekf_.x_ = VectorXd(4);

    // Convert radar from polar to cartesian coordinates and compose object state vector
    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      float ro = measurement_pack.raw_measurements_(0);
      float theta = measurement_pack.raw_measurements_(1);
      float ro_dot = measurement_pack.raw_measurements_(2);

      ekf_.x_ << ro * cos(theta),
              ro * sin(theta),
              ro_dot * cos(theta),
              ro_dot * sin(theta);

    // Compose simple vector from laser measurements
    } else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      ekf_.x_ << measurement_pack.raw_measurements_(0),
              measurement_pack.raw_measurements_(1),
              0,
              0;
    }

    // Init object covariance matrix
    ekf_.P_ = MatrixXd(4, 4);
    ekf_.P_ << 1, 0, 0,    0,
               0, 1, 0,    0,
               0, 0, 1000, 0,
               0, 0, 0,    1000;

    // Init state transition matrix. Same for radar and laser as for now
    ekf_.F_ = MatrixXd::Identity(4, 4);

    previous_timestamp_ = measurement_pack.timestamp_;

    // No need to predict or update on initialisation step
    is_initialized_ = true;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  // Calculate time delta for further processing in seconds
  float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;

  // Save timestamp for the next step
  previous_timestamp_ = measurement_pack.timestamp_;

  // Update state transition matrix with time delta
  ekf_.F_(0, 2) = dt;
  ekf_.F_(1, 3) = dt;

  // set process covariance matrix
  ekf_.Q_ = this->CalculateQ(dt);

  ekf_.Predict();

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  // Extended Calman Filter Update for Radar
  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    ekf_.H_ = tools.CalculateJacobian(ekf_.x_);
    ekf_.R_ = R_radar_;
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);

  // Regular Calman Filter Update for Laser
  } else {
    ekf_.H_ = H_laser_;
    ekf_.R_ = R_laser_;
    ekf_.UpdateRegular(measurement_pack.raw_measurements_);
  }
}

MatrixXd FusionEKF::CalculateQ(float dt) {
  float dt2_ax = pow(dt, 2)*Q_noise_ax_;
  float dt2_ay = pow(dt, 2)*Q_noise_ay_;
  float dt3_ax = dt*dt2_ax/2;
  float dt3_ay = dt*dt2_ay/2;
  float dt4_ax = dt*dt3_ax/2;
  float dt4_ay = dt*dt3_ay/2;

  MatrixXd Q_ = MatrixXd(4, 4);
  Q_ << dt4_ax, 0,      dt3_ax, 0,
        0,      dt4_ay, 0,      dt3_ay,
        dt3_ax, 0,      dt2_ax, 0,
        0,      dt3_ay, 0,      dt2_ay;

  return Q_;
}