class TrackableObject:
    def __init__(self, objectID, centroid):
        #store the object ID, then initialize a list of centroids
        #using the current centroid
        self.objectID = objectID
        self.centroids = [centroid]

        #initialize a boolean used to indicate if the objective has
        #already been counted of not
        self.counted = False
