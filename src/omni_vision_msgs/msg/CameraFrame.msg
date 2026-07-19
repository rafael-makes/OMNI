# One camera's most recent frame, JPEG-encoded.
#
# Exists because a service response cannot hold an array of uint8[] directly —
# "give me every camera at once" (GetCameraFrame with camera_id "all") needs a
# message type to nest. Also carries camera_id so a multi-camera reply is
# self-describing: the caller must be able to tell which view is which without
# relying on array order, since the fusion prompt names the cameras.
string camera_id           # "head", "rear", ...
uint8[] jpeg               # JPEG bytes
uint32 width               # pixels; 0 if unknown (frame_server does not decode)
uint32 height
builtin_interfaces/Time stamp   # capture time of the frame, not reply time
float32 age_seconds        # how old this frame was when served
