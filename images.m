outputFolder = fullfile('FOOD');
rootFolder=fullfile(outputFolder,'caltech');
categories={'Burger','cake','dosa','idli','Frankie','Momos','pizza','Pohe','Samosa','Sandwich','Vadapav'};
imds=imageDatastore(fullfile(rootFolder,categories),'LabelSource','foldernames');
tbl = countEachLabel(imds);
minSetCount = min(tbl{:,2});
maxNumImages=5;
minSetCount=min(maxNumImages,minSetCount);
imds=splitEachLabel(imds,minSetCount,'randomize');
countEachLabel(imds);
Burger=find(imds.Labels=='Burger',1);
cake=find(imds.Labels=='cake',1);
dosa=find(imds.Labels=='dosa',1);
idli=find(imds.Labels=='idli',1);
Frankie=find(imds.Labels=='Frankie',1);
Momos=find(imds.Labels=='Momos',1);
pizza=find(imds.Labels=='pizza',1);
Pohe=find(imds.Labels=='Pohe',1);
Samosa=find(imds.Labels=='Samosa',1);
Sandwich=find(imds.Labels=='Sandwich',1);
Vadapav=find(imds.Labels=='Vadapav',1);





net=resnet50();

set(gca,'YLim',[150 170]);

net.Layers(1)
net.Layers(end)

numel(net.Layers(end).ClassNames)
[trainingSet,testSet] = splitEachLabel(imds,0.5,'randomize');

imageSize = net.Layers(1).InputSize;

augmentedTrainingSet = augmentedImageDatastore(imageSize,trainingSet,'ColorPreprocessing','gray2rgb');

augmentedTestSet = augmentedImageDatastore(imageSize,testSet,'ColorPreprocessing','gray2rgb');

w1 = net.Layers(2).Weights;
w1 = mat2gray(w1);
w1 = imresize(w1,5);

featureLayer = 'fc1000';
trainingFeatures = activations(net,augmentedTrainingSet,featureLayer,...
'MiniBatchSize',32,'OutputAs','columns');

trainingLabels = trainingSet.Labels;
classifier = fitcecoc(trainingFeatures,trainingLabels,...
'Learner','Linear','Coding','onevsall','ObservationsIn','columns');

testFeatures = activations(net,augmentedTestSet,featureLayer,...
'MiniBatchSize',32,'OutputAs','columns');

predictLabels = predict(classifier,testFeatures,'ObservationsIn','columns');

testLabels = testSet.Labels;
confMat = confusionmat(testLabels,predictLabels)
confMat = bsxfun(@rdivide,confMat,sum(confMat,2))

mean(diag(confMat))
[fname, path]=uigetfile('.png','provide an Image for testing');
fname=strcat(path, fname);
newImage =imread(fname);


ds = augmentedImageDatastore(imageSize,newImage,'ColorPreprocessing','gray2rgb');

imageFeatures = activations(net,ds,featureLayer,'MiniBatchSize',32,'OutputAs','columns');

label = predict(classifier,imageFeatures,'ObservationsIn','columns');
sprintf('The loaded image belongs to %s class',label)