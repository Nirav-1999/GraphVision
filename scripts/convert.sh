#!/bin/bash
#convert
for image in test_set/histogram/*.gif; do
        convert  "$image"  "${image%.gif}.jpg"
        echo “image $image converted to ${image%.gif}.jpg ”
done
exit 0 
