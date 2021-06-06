outputFolder = fullfile('FOOD');
rootFolder=fullfile(outputFolder,'Delhi');
categories={'alootikki','dahibhale','dalautkechaat','nankhati','Paneertikka','rolls'};
imds=imageDatastore(fullfile(rootFolder,categories),'LabelSource','foldernames');
tbl = countEachLabel(imds)
minSetCount = min(tbl{:,2})
imds=splitEachLabel(imds,minSetCount,'randomize');
countEachLabel(imds);
alootikki=find(imds.Labels=='alootikki',1);
dahibhale=find(imds.Labels=='dahibhale',1);
dalautkechaat=find(imds.Labels=='dalautkechaat',1);
nankhati=find(imds.Labels=='nankhati',1);
Paneertikka=find(imds.Labels=='Paneertikka',1);
rolls=find(imds.Labels=='rolls',1);





net=resnet50();

set(gca,'YLim',[150 170]);

net.Layers(1)
net.Layers(end)

numel(net.Layers(end).ClassNames)
[trainingSet,testSet] = splitEachLabel(imds,0.3,'randomize');

imageSize = net.Layers(1).InputSize;

augmentedTrainingSet = augmentedImageDatastore(imageSize,trainingSet,'ColorPreprocessing','gray2rgb')

augmentedTestSet = augmentedImageDatastore(imageSize,testSet,'ColorPreprocessing','gray2rgb')

w1 = net.Layers(2).Weights;
w1 = mat2gray(w1);
w1 = imresize(w1,5); 

featureLayer = 'fc1000';
trainingFeatures = activations(net,augmentedTrainingSet,featureLayer,'MiniBatchSize',32,'OutputAs','columns');

trainingLabels = trainingSet.Labels;
classifier = fitcecoc(trainingFeatures,trainingLabels, 'Learners','Linear','Coding','onevsall','ObservationsIn','columns');

testFeatures = activations(net,augmentedTestSet,featureLayer,'MiniBatchSize',32,'OutputAs','columns');

predictLabels = predict(classifier,testFeatures,'ObservationsIn','columns');

testLabels = testSet.Labels;
confMat = confusionmat(testLabels,predictLabels);
confMat = bsxfun(@rdivide,confMat,sum(confMat,2))

mean(diag(confMat))
[fname, path]=uigetfile('.png','provide an Image for testing');
fname=strcat(path, fname);
newImage =imread(fname);



ds = augmentedImageDatastore(imageSize,newImage,'ColorPreprocessing','gray2rgb');

imageFeatures = activations(net,ds,featureLayer,'MiniBatchSize',32,'OutputAs','columns');

label = predict(classifier,imageFeatures,'ObservationsIn','columns');
sprintf('The loaded image belongs to %s class',label)
