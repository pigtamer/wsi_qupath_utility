//This is actually two scripts, the first should be run in the image you are copying from
//and the second in the image you are copying to.
// 本文件用于将标注导出到文本文件或者从文本文件将标注导入到 Qupath 和 WSI 数据文件

// Execute this file in Qupath
// 注意本文件在 qupath 内执行

def MODE = "EXPORT" // export annotations to text file by default

if(MODE != "IMPORT") {
//Export section
    def path = buildFilePath(PROJECT_BASE_DIR, 'annotations-' + getProjectEntry().getImageName() + '.txt')

    def annotations = getAnnotationObjects().collect { new qupath.lib.objects.PathAnnotationObject(it.getROI(), it.getPathClass()) }
    
    new File(path).withObjectOutputStream {
        it.writeObject(annotations)
    }
    print 'Done!'
}else {
//There should now be a saved .txt file in the base project directory that holds the annotation information.
//Swap images and run the second set of code using the file name generated in the first half of the code.

// Current path works for an image that has the same name as the original image, but you can adjust that.
    clearAllObjects()
    
    def path = buildFilePath(PROJECT_BASE_DIR, 'annotations-' + getProjectEntry().getImageName() + '.txt')
    def annotations = null
    new File(path).withObjectInputStream {
        annotations = it.readObject()
    }
    addObjects(annotations)
    getAnnotationObjects().each { it.setLocked(true) }
    print 'Added ' + annotations
}