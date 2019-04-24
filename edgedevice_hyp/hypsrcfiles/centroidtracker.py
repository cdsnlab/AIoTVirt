# import the necessary packages
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
import queue
import streamQueue

class CentroidTracker:
        def __init__(self, maxDisappeared=50, maxDistance=50, queuesize = 10):
                # queuesize --> how many centroid locations are we going to look at?

                # initialize the next unique object ID along with two ordered
                # dictionaries used to keep track of mapping a given object
                # ID to its centroid and number of consecutive frames it has
                # been marked as "disappeared", respectively
                self.nextObjectID = 0
                self.objects = OrderedDict() # only stores centroids 
                self.objectsrect = OrderedDict()
                self.disappeared = OrderedDict()

                # store the number of maximum consecutive frames a given
                # object is allowed to be marked as "disappeared" until we
                # need to deregister the object from tracking
                self.maxDisappeared = maxDisappeared

                # store the maximum distance between centroids to associate
                # an object -- if the distance is larger than this maximum
                # distance we'll start to mark the object as "disappeared"
                self.maxDistance = maxDistance
                
                # store the actual index of existing objects 
                self.actualindex = []
                self.queuesize = queuesize
                self.lqx = {}
                self.lqy = {}
 

        def getall(self):
                # prints all existing object indexs 
                for i in (self.actualindex):
                        print("Object ID:", i, self.objects[i])
                        print("Object ID:", i, self.objectsrect[i])

        def checknumberofexisting(self):
                return (len(self.actualindex))


        def get_object_rect_by_id (self, objectID):
                return self.objectsrect[objectID]

        def register(self, centroid):
                # when registering an object we use the next available object
                # ID to store the centroid
                self.objects[self.nextObjectID] = centroid
                self.actualindex.append(self.nextObjectID)
                self.disappeared[self.nextObjectID] = 0
                self.lqx[self.nextObjectID] = (streamQueue.streamQueue(self.queuesize))
                self.lqy[self.nextObjectID] = (streamQueue.streamQueue(self.queuesize))
                self.lqx[self.nextObjectID].enqueue(centroid[0])
                self.lqy[self.nextObjectID].enqueue(centroid[1])


        def predict(self, objectID, next_spot=30):
                #predict next x,y of this object after next_spot number of frames
                xsize = self.lqx[objectID].queue.qsize()
                ysize = self.lqy[objectID].queue.qsize()
                
                disx = self.lqx[objectID].queue.queue[xsize-1] - self.lqx[objectID].queue.queue[0]
                disy = self.lqy[objectID].queue.queue[ysize-1] - self.lqy[objectID].queue.queue[0]

                #print((disx / self.queuesize * next_spot) + self.lqx[objectID].queue.queue[xsize-1])
                #print((disy / self.queuesize * next_spot) + self.lqy[objectID].queue.queue[ysize-1])
                return (((disx / self.queuesize * next_spot) + self.lqx[objectID].queue.queue[xsize-1]), ((disy / self.queuesize * next_spot) + self.lqy[objectID].queue.queue[ysize-1]))
                #print(self.lqx[objectID].queue.queue[0])
                #print(self.lqy[objectID].queue.queue[0])


        def register_rects(self, rects):
                self.objectsrect[self.nextObjectID] = rects
                print("registering new RECTS :", self.nextObjectID, rects)

                self.nextObjectID += 1


        def deregister(self, objectID):
                # to deregister an object ID we delete the object ID from
                # both of our respective dictionaries
                del self.objects[objectID]                
                del self.disappeared[objectID]
                del self.objectsrect[objectID]
                self.actualindex.remove(objectID)
                del self.lqx[objectID]
                del self.lqy[objectID]

        def update(self, rects):
                # check to see if the list of input bounding box rectangles
                # is empty
                if len(rects) == 0:
                        # loop over any existing tracked objects and mark them
                        # as disappeared
                        for objectID in list(self.disappeared.keys()):
                                self.disappeared[objectID] += 1

                                # if we have reached a maximum number of consecutive
                                # frames where a given object has been marked as
                                # missing, deregister it
                                if self.disappeared[objectID] > self.maxDisappeared:
                                        self.deregister(objectID)

                        # return early as there are no centroids or tracking info
                        # to update
                        return self.objects

                # initialize an array of input centroids for the current frame
                inputCentroids = np.zeros((len(rects), 2), dtype="int")

                # loop over the bounding box rectangles
                for (i, (startX, startY, endX, endY)) in enumerate(rects):
                        # use the bounding box coordinates to derive the centroid
                        cX = int((startX + endX) / 2.0)
                        cY = int((startY + endY) / 2.0)
                        inputCentroids[i] = (cX, cY)

                # if we are currently not tracking any objects take the input
                # centroids and register each of them
                if len(self.objects) == 0:
                        for i in range(0, len(inputCentroids)):
                                self.register(inputCentroids[i])
                                self.register_rects(rects)
                                

                # otherwise, are are currently tracking objects so we need to
                # try to match the input centroids to existing object
                # centroids
                else:
                        # grab the set of object IDs and corresponding centroids
                        objectIDs = list(self.objects.keys())
                        objectCentroids = list(self.objects.values())
                        objectRects = list(self.objectsrect.values())

                        # compute the distance between each pair of object
                        # centroids and input centroids, respectively -- our
                        # goal will be to match an input centroid to an existing
                        # object centroid
                        D = dist.cdist(np.array(objectCentroids), inputCentroids)

                        # in order to perform this matching we must (1) find the
                        # smallest value in each row and then (2) sort the row
                        # indexes based on their minimum values so that the row
                        # with the smallest value as at the *front* of the index
                        # list
                        rows = D.min(axis=1).argsort()

                        # next, we perform a similar process on the columns by
                        # finding the smallest value in each column and then
                        # sorting using the previously computed row index list
                        cols = D.argmin(axis=1)[rows]

                        #print (rows, cols)

                        # in order to determine if we need to update, register,
                        # or deregister an object we need to keep track of which
                        # of the rows and column indexes we have already examined
                        usedRows = set()
                        usedCols = set()

                        # loop over the combination of the (row, column) index
                        # tuples
                        for (row, col) in zip(rows, cols):
                                # if we have already examined either the row or
                                # column value before, ignore it
                                if row in usedRows or col in usedCols:
                                        continue

                                # if the distance between centroids is greater than
                                # the maximum distance, do not associate the two
                                # centroids to the same object
                                #print(D[row,col])
                                if D[row, col] > self.maxDistance:
                                        continue

                                # otherwise, grab the object ID for the current row,
                                # set its new centroid, and reset the disappeared
                                # counter
                                objectID = objectIDs[row]
                                self.objects[objectID] = inputCentroids[col]
                                #print(inputCentroids[col][0])
                                self.objectsrect[objectID] = rects[col]
                                self.disappeared[objectID] = 0
                                self.lqx[objectID].enqueue(inputCentroids[col][0])
                                self.lqy[objectID].enqueue(inputCentroids[col][1])

                                # indicate that we have examined each of the row and
                                # column indexes, respectively
                                usedRows.add(row)
                                usedCols.add(col)

                        # compute both the row and column index we have NOT yet
                        # examined
                        unusedRows = set(range(0, D.shape[0])).difference(usedRows)
                        unusedCols = set(range(0, D.shape[1])).difference(usedCols)

                        # in the event that the number of object centroids is
                        # equal or greater than the number of input centroids
                        # we need to check and see if some of these objects have
                        # potentially disappeared
                        if D.shape[0] >= D.shape[1]:
                                # loop over the unused row indexes
                                for row in unusedRows:
                                        # grab the object ID for the corresponding row
                                        # index and increment the disappeared counter
                                        objectID = objectIDs[row]
                                        self.disappeared[objectID] += 1

                                        # check to see if the number of consecutive
                                        # frames the object has been marked "disappeared"
                                        # for warrants deregistering the object
                                        if self.disappeared[objectID] > self.maxDisappeared:
                                                self.deregister(objectID)

                        # otherwise, if the number of input centroids is greater
                        # than the number of existing object centroids we need to
                        # register each new input centroid as a trackable object
                        else:
                                for col in unusedCols:
                                        self.register(inputCentroids[col])
                                        self.register_rects(rects[col])
                                        
                                        

                                        

                # return the set of trackable objects
                return self.objects
