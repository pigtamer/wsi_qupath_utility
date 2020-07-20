/**
 * Script to transfer QuPath objects from one image to another, applying an AffineTransform to any ROIs.

 More info on custom scripting:
 https://gist.github.com/Svidro/5829ba53f927e79bb6e370a6a6747cfd

 */

import qupath.lib.objects.PathCellObject
import qupath.lib.objects.PathDetectionObject
import qupath.lib.objects.PathObject
import qupath.lib.objects.PathObjects
import qupath.lib.objects.PathTileObject
import qupath.lib.roi.RoiTools
import qupath.lib.roi.interfaces.ROI

import java.awt.geom.AffineTransform

import static qupath.lib.gui.scripting.QPEx.*


// Transformation matrix
// Annotate on IHC, run on HE.
// Trans. Mat. from HE to IHC. (Align with HE and open IHC)


def he2ihcTransMatrices = [
        '01_17-7885_Ki67_': [1.0000, 0.0000, -4110.1113,
                             0.0000, 1.0000, -5672.8013], //ok
        '01_17-8107_Ki67_': [1.0000, 0.0000, -3619.2339,
                             -0.0000, 1.0000, -39.6530], //ok
        '01_14-3768_Ki67_': [1.0000, 0.0000, -4099.9741,
                             0.0000, 1.0000, -51.2197],//ok
        '01_14-7015_Ki67_': [1.0000, 0.0000, -3919.6276,
                             0.0000, 1.0000, 37.3973], //ok
        '01_15-1052_Ki67_': [1.0000, -0.0000, 860.3158,
                             0.0000, 1.0000, -1432.6411], //ok
        '01_15-2502_Ki67_': [1.0000, 0.0000, -1052.4520,
                             0.0000, 1.0000, -15.0699], //ok
        '01_17-5256_Ki67_': [1.0000, 0.0000, -5321.4238,
                             0.0000, 1.0000, -3674.8283], //ok
        '01_17-6747_Ki67_': [1.0000, 0.0000, 279.7157,
                             0.0000, 1.0000, -1757.3694], //ok
        '01_17-7930_Ki67_': [1.0000, 0.0000, -606.0000,
                             0.0000, 1.0000, 1468.9137] //ok
]

def displayedNamesList = ['Tumor', 'Blood', 'Healthy Tissue']

// SET ME! Delete existing objects
def deleteExisting = true

// SET ME! Change this if things end up in the wrong place
def createInverse = true

// Process all HE images in the project.
for (WSI_ID in he2ihcTransMatrices.keySet()) {
    def matrix = he2ihcTransMatrices[WSI_ID]

// Define image containing the original objects (must be in the current project)
    def ihcAnnotImgName = WSI_ID + "IHC.ndpi"
    def heTargetImgName = WSI_ID + "HE.ndpi"

// Get the project & the requested image name
    def project = getProject()

    def he_entry = project.getImageList().find { it.getImageName() == heTargetImgName }
    def ihc_entry = project.getImageList().find { it.getImageName() == ihcAnnotImgName }
    if (ihc_entry == null) {
        print 'Could not find image with name ' + ihcAnnotImgName
        return
    }
    if (he_entry == null) {
        print 'Could not find image with name ' + heTargetImgName
        return
    }


    def heData = he_entry.readImageData()
    def heHierarchy = heData.getHierarchy()

    def ihcHierarchy = ihc_entry.readHierarchy()
    def ihcObjects = ihcHierarchy.getRootObject().getChildObjects()

    print ihcObjects
// Define the transformation matrix
    def transform = new AffineTransform(
            matrix[0] as double, matrix[3] as double, matrix[1] as double,
            matrix[4] as double, matrix[2] as double, matrix[5] as double
    )
    if (createInverse) {
        transform = transform.createInverse()
    }

    if (deleteExisting) {
        heHierarchy.clearAll()
    }

    def newObjects = []
    for (pathObject in ihcObjects) {
        if (displayedNamesList.contains(pathObject.getProperties().get('displayedName'))) {
            print(pathObject.getProperties().get('displayedName'))
            newObjects << transformObject(pathObject, transform)
        }
    }
//    addObjects(newObjects)
    heHierarchy.addPathObjects(newObjects)
    he_entry.saveImageData(heData)
    print(ihcAnnotImgName + " processed.")
}

print 'Done!'

/**
 * Transform object, recursively transforming all child objects
 *
 * @param pathObject
 * @param transform
 * @return
 */
PathObject transformObject(PathObject pathObject, AffineTransform transform) {
    // Create a new object with the converted ROI
    def roi = pathObject.getROI()
    def roi2 = transformROI(roi, transform)
    def newObject = null
    if (pathObject instanceof PathCellObject) {
        def nucleusROI = pathObject.getNucleusROI()
        if (nucleusROI == null) {
            newObject = PathObjects.createCellObject(roi2, pathObject.getPathClass(), pathObject.getMeasurementList())
        } else {
            newObject = PathObjects.createCellObject(roi2, transformROI(nucleusROI, transform), pathObject.getPathClass(), pathObject.getMeasurementList())
        }
    } else if (pathObject instanceof PathTileObject) {
        newObject = PathObjects.createTileObject(roi2, pathObject.getPathClass(), pathObject.getMeasurementList())
    } else if (pathObject instanceof PathDetectionObject) {
        newObject = PathObjects.createDetectionObject(roi2, pathObject.getPathClass(), pathObject.getMeasurementList())
    } else {
        newObject = PathObjects.createAnnotationObject(roi2, pathObject.getPathClass(), pathObject.getMeasurementList())
    }
    // Handle child objects
    if (pathObject.hasChildren()) {
        newObject.addPathObjects(pathObject.getChildObjects().collect({ transformObject(it, transform) }))
    }
    return newObject
}

/**
 * Transform ROI (via conversion to Java AWT shape)
 *
 * @param roi
 * @param transform
 * @return
 */
ROI transformROI(ROI roi, AffineTransform transform) {
    def shape = RoiTools.getShape(roi)
    // Should be able to use roi.getShape() - but there's currently a bug in it for rectangles/ellipses!
    shape2 = transform.createTransformedShape(shape)
    return RoiTools.getShapeROI(shape2, roi.getImagePlane(), 0.5)
}


def tile_size = 463.4624 //um