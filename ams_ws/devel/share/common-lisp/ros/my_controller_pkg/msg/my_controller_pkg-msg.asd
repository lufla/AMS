
(cl:in-package :asdf)

(defsystem "my_controller_pkg-msg"
  :depends-on (:roslisp-msg-protocol :roslisp-utils )
  :components ((:file "_package")
    (:file "THK_AMS_msg1" :depends-on ("_package_THK_AMS_msg1"))
    (:file "_package_THK_AMS_msg1" :depends-on ("_package"))
  ))