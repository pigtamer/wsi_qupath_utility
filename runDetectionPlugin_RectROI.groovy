

//Make sure you replace the default cell (or whatever) detection script with your own.


import qupath.lib.gui.QuPathGUI
import qupath.lib.roi.RectangleROI
import qupath.lib.objects.PathAnnotationObject
import qupath.lib.objects.classes.PathClassFactory

        clearAllObjects()


//Use either "project" OR "outputFolder" to determine where your detection files will go
def project = QuPathGUI.getInstance().getProject().getBaseDirectory()
project = project.toString()+"\\detectionMeasurements\\"

//Make sure the output folder exists
// mkdirs(project)
//def outputFolder = "D:\\Results\\"
//mkdirs(outputFolder)
sizePixels = 2048
// Create a new Rectangle ROI
def roi = new RectangleROI(0, 0, sizePixels, sizePixels)
print(roi)
// Create & new annotation & add it to the object hierarchy
def annot = new PathAnnotationObject(roi, PathClassFactory.getPathClass("Region"))
def imageData = getCurrentImageData()

addObjects(annot)

hierarchy = getCurrentHierarchy()
def annotations = getAnnotationObjects()
int i = 1
clearDetections()  //Just in case, so that the first detection list won't end up with extra stuff that was laying around

for (annotation in annotations)
{
    if (annotation.getProperties().get('displayedName') != "Kimura_sample"){
        hierarchy.getSelectionModel().clearSelection();
        selectObjects{p -> p == annotation}
        //**********************************************************************
        //REPLACE THIS PART WITH YOUR OWN DETECTION CREATION SCRIPT

        runPlugin('qupath.imagej.detect.cells.PositiveCellDetection', '{"detectionImageBrightfield": "Optical Density Sum"}');

        // runPlugin('qupath.imagej.detect.nuclei.PositiveCellDetection', '{"detectionImageBrightfield": "Hematoxylin OD",  "requestedPixelSizeMicrons": 0.5,  "backgroundRadiusMicrons": 8.0,  "medianRadiusMicrons": 0.0,  "sigmaMicrons": 1.5,  "minAreaMicrons": 10.0,  "maxAreaMicrons": 400.0,  "threshold": 0.02,  "maxBackground": 2.0,  "watershedPostProcess": true,  "excludeDAB": false,  "cellExpansionMicrons": 5.0,  "includeNuclei": true,  "smoothBoundaries": true,  "makeMeasurements": true, "thresholdCompartment": "Nucleus: DAB OD max"}');
        //************************************************************************

        // saveDetectionMeasurements(project+" "+i+"detections.txt",)
        i+=1
        // clearDetections()
    }
}

//Potentially replace all of the detections for viewing, after finishing the export

//selectAnnotations()
//runPlugin('qupath.imagej.detect.nuclei.WatershedCellDetection', '{"detectionImageBrightfield": "Hematoxylin OD",  "requestedPixelSizeMicrons": 0.5,  "backgroundRadiusMicrons": 8.0,  "medianRadiusMicrons": 0.0,  "sigmaMicrons": 1.5,  "minAreaMicrons": 10.0,  "maxAreaMicrons": 400.0,  "threshold": 0.1,  "maxBackground": 2.0,  "watershedPostProcess": true,  "excludeDAB": false,  "cellExpansionMicrons": 5.0,  "includeNuclei": true,  "smoothBoundaries": true,  "makeMeasurements": true}');