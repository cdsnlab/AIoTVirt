# 2018-03-09

I made some minor modifications to the imutils package to be able to change
the name of the camera thread in order to track more easily the resource
usage of each thread. You can copy this into wherever imutils.video is installed.
It's probably /usr/local/lib/python3.5/site-packages/imutils/video. For
debian it's sometimes called dist-packages instead of site-packages.

Alternatively simply remove the "name" parameter when initializing a
VideoStream object and it should work normally.
