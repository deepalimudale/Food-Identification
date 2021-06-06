outputFolder = fullfile('FOOD');
rootFolder=fullfile(outputFolder,'MAHARASHTRA');
categories={'Pavbhaji','Pithlabhakri','POHE','Puranpoli','Sabudana','Shreekhand','VADAPAV'};
imds=imageDatastore(fullfile(rootFolder,categories),'LabelSource','foldernames');
tbl = countEachLabel(imds)
minSetCount = min(tbl{:,2})
imds=splitEachLabel(imds,minSetCount,'randomize');
countEachLabel(imds);
Pavbhaji=find(imds.Labels=='Pavbhaji',1);
Pithlabhakri=find(imds.Labels=='Pithlabhakri',1);
POHE=find(imds.Labels=='POHE',1);
Puranpoli=find(imds.Labels=='Puranpoli',1);
Sabudana=find(imds.Labels=='Sabudana',1);
Shreekhand=find(imds.Labels=='Shreekhand',1);
VADAPAV=find(imds.Labels=='VADAPAV',1);

figure
subplot(2,4,1);
imshow(readimage(imds,Pavbhaji));
subplot(2,4,2);
imshow(readimage(imds,Pithlabhakri));
subplot(2,4,3);
imshow(readimage(imds,POHE));
subplot(2,4,4);
imshow(readimage(imds,Puranpoli));
subplot(2,4,5);
imshow(readimage(imds,Sabudana));
subplot(2,4,6);
imshow(readimage(imds,Shreekhand));
subplot(2,4,7);
imshow(readimage(imds,VADAPAV));





net=resnet50();
figure
plot(net)
title('Architecture of ResNet-50')
set(gca,'YLim',[150 170]);

net.Layers(1)
net.Layers(end)

numel(net.Layers(end).ClassNames)
[trainingSet,testSet] = splitEachLabel(imds,0.5,'randomize');

imageSize = net.Layers(1).InputSize;

augmentedTrainingSet = augmentedImageDatastore(imageSize,trainingSet,'ColorPreprocessing','gray2rgb')

augmentedTestSet = augmentedImageDatastore(imageSize,testSet,'ColorPreprocessing','gray2rgb')

w1 = net.Layers(2).Weights;
w1 = mat2gray(w1);
w1 = imresize(w1,5); 
figure
montage(w1)
title('First Convolutional Layer Weights')

featureLayer = 'fc1000';
trainingFeatures = activations(net,augmentedTrainingSet,featureLayer,'MiniBatchSize',64,'OutputAs','columns');

trainingLabels = trainingSet.Labels;
classifier = fitcecoc(trainingFeatures,trainingLabels, 'Learners','Linear','Coding','onevsall','ObservationsIn','columns');

testFeatures = activations(net,augmentedTestSet,featureLayer,'MiniBatchSize',64,'OutputAs','columns');

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
