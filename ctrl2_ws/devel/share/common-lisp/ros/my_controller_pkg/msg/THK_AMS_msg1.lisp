; Auto-generated. Do not edit!


(cl:in-package my_controller_pkg-msg)


;//! \htmlinclude THK_AMS_msg1.msg.html

(cl:defclass <THK_AMS_msg1> (roslisp-msg-protocol:ros-message)
  ((x
    :reader x
    :initarg :x
    :type cl:float
    :initform 0.0)
   (y
    :reader y
    :initarg :y
    :type cl:float
    :initform 0.0)
   (angle
    :reader angle
    :initarg :angle
    :type cl:float
    :initform 0.0))
)

(cl:defclass THK_AMS_msg1 (<THK_AMS_msg1>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <THK_AMS_msg1>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'THK_AMS_msg1)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name my_controller_pkg-msg:<THK_AMS_msg1> is deprecated: use my_controller_pkg-msg:THK_AMS_msg1 instead.")))

(cl:ensure-generic-function 'x-val :lambda-list '(m))
(cl:defmethod x-val ((m <THK_AMS_msg1>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader my_controller_pkg-msg:x-val is deprecated.  Use my_controller_pkg-msg:x instead.")
  (x m))

(cl:ensure-generic-function 'y-val :lambda-list '(m))
(cl:defmethod y-val ((m <THK_AMS_msg1>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader my_controller_pkg-msg:y-val is deprecated.  Use my_controller_pkg-msg:y instead.")
  (y m))

(cl:ensure-generic-function 'angle-val :lambda-list '(m))
(cl:defmethod angle-val ((m <THK_AMS_msg1>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader my_controller_pkg-msg:angle-val is deprecated.  Use my_controller_pkg-msg:angle instead.")
  (angle m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <THK_AMS_msg1>) ostream)
  "Serializes a message object of type '<THK_AMS_msg1>"
  (cl:let ((bits (roslisp-utils:encode-double-float-bits (cl:slot-value msg 'x))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-double-float-bits (cl:slot-value msg 'y))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-double-float-bits (cl:slot-value msg 'angle))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) bits) ostream))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <THK_AMS_msg1>) istream)
  "Deserializes a message object of type '<THK_AMS_msg1>"
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'x) (roslisp-utils:decode-double-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'y) (roslisp-utils:decode-double-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'angle) (roslisp-utils:decode-double-float-bits bits)))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<THK_AMS_msg1>)))
  "Returns string type for a message object of type '<THK_AMS_msg1>"
  "my_controller_pkg/THK_AMS_msg1")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'THK_AMS_msg1)))
  "Returns string type for a message object of type 'THK_AMS_msg1"
  "my_controller_pkg/THK_AMS_msg1")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<THK_AMS_msg1>)))
  "Returns md5sum for a message object of type '<THK_AMS_msg1>"
  "57832a67d2f8a00310788a06c92c59b2")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'THK_AMS_msg1)))
  "Returns md5sum for a message object of type 'THK_AMS_msg1"
  "57832a67d2f8a00310788a06c92c59b2")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<THK_AMS_msg1>)))
  "Returns full string definition for message of type '<THK_AMS_msg1>"
  (cl:format cl:nil "# Coordinates~%~%float64 x~%float64 y~%float64 angle~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'THK_AMS_msg1)))
  "Returns full string definition for message of type 'THK_AMS_msg1"
  (cl:format cl:nil "# Coordinates~%~%float64 x~%float64 y~%float64 angle~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <THK_AMS_msg1>))
  (cl:+ 0
     8
     8
     8
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <THK_AMS_msg1>))
  "Converts a ROS message object to a list"
  (cl:list 'THK_AMS_msg1
    (cl:cons ':x (x msg))
    (cl:cons ':y (y msg))
    (cl:cons ':angle (angle msg))
))
