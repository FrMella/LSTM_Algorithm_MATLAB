load WaveformData
data(1:5)

numChannels = size(data{1},1)

figure
tiledlayout(2,2)
for i = 1:4
    nexttile
    stackedplot(data{i}',DisplayLabels="Channel " + (1:numChannels));
    title("Observation " + i)
    xlabel("Time Step")
end



options = trainingOptions("adam", ...
    MaxEpochs=200, ...
    SequencePaddingDirection="left", ...
    Shuffle="every-epoch", ...
    Plots="training-progress", ...
    Verbose=0);

numObservations = numel(data);
XTrain = data(1:floor(0.9*numObservations));
XValidation = data(floor(0.9*numObservations)+1:end);

numDownsamples = 2;

sequenceLengths = zeros(1,numel(XTrain));

for n = 1:numel(XTrain)
    X = XTrain{n};
    cropping = mod(size(X,2), 2^numDownsamples);
    X(:,end-cropping+1:end) = [];
    XTrain{n} = X;
    sequenceLengths(n) = size(X,2);
end

for n = 1:numel(XValidation)
    X = XValidation{n};
    cropping = mod(size(X,2),2^numDownsamples);
    X(:,end-cropping+1:end) = [];
    XValidation{n} = X;
end

minLength = min(sequenceLengths);
filterSize = 7;
numFilters = 16;
dropoutProb = 0.2;

layers = sequenceInputLayer(numChannels,Normalization="zscore",MinLength=minLength);

for i = 1:numDownsamples
    layers = [
        layers
        convolution1dLayer(filterSize,(numDownsamples+1-i)*numFilters,Padding="same",Stride=2)
        reluLayer
        dropoutLayer(dropoutProb)];
end

for i = 1:numDownsamples
    layers = [
        layers
        transposedConv1dLayer(filterSize,i*numFilters,Cropping="same",Stride=2)
        reluLayer
        dropoutLayer(dropoutProb)];
end

layers = [
    layers
    transposedConv1dLayer(filterSize,numChannels,Cropping="same")
    regressionLayer];

net = trainNetwork(XTrain,XTrain,layers,options);

YValidation = predict(net,XValidation);

MAEValidation = zeros(numel(XValidation),1);
for n = 1:numel(XValidation)
    X = XValidation{n};
    Y = YValidation{n};
    MAEValidation(n) = mean(abs(Y - X),'all');
end

figure
histogram(MAEValidation)
xlabel("Mean Absolute Error (MAE)")
ylabel("Frequency")
title("Representative Samples")

MAEbaseline = max(MAEValidation)

XNew = XValidation;


numAnomalousSequences = 20;
idx = randperm(numel(XValidation),numAnomalousSequences);

for i = 1:numAnomalousSequences
    X = XNew{idx(i)};

    idxPatch = 50:60;
    XPatch = X(:,idxPatch);
    X(:,idxPatch) = 4*abs(XPatch);

    XNew{idx(i)} = X;
end

YNew = predict(net,XNew);


MAENew = zeros(numel(XNew),1);
for n = 1:numel(XNew)
    X = XNew{n};
    Y = YNew{n};
    MAENew(n) = mean(abs(Y - X),'all');
end

figure
histogram(MAENew)
xlabel("Mean Absolute Error (MAE)")
ylabel("Frequency")
title("New Samples")
hold on
xline(MAEbaseline,"r--")
legend(["Data" "Baseline MAE"])

[~,idxTop] = sort(MAENew,"descend");
idxTop(1:10)

X = XNew{idxTop(1)};
Y = YNew{idxTop(1)};

figure
t = tiledlayout(numChannels,1);
title(t,"Sequence " + idxTop(1))

for i = 1:numChannels
    nexttile

    plot(X(i,:))
    box off
    ylabel("Channel " + i)

    hold on
    plot(Y(i,:),"--")
end

nexttile(1)
legend(["Original" "Reconstructed"])

MAE = mean(abs(Y - X),1);

windowSize = 7;
thr = 1.1*MAEbaseline;

idxAnomaly = false(1,size(X,2));
for t = 1:(size(X,2) - windowSize + 1)
    idxWindow = t:(t + windowSize - 1);

    if all(MAE(idxWindow) > thr)
        idxAnomaly(idxWindow) = true;
    end
end

figure
t = tiledlayout(numChannels,1);
title(t,"Anomaly Detection ")

for i = 1:numChannels
    nexttile
    plot(X(i,:));
    ylabel("Channel " + i)
    box off
    hold on

    XAnomalous = nan(1,size(X,2));
    XAnomalous(idxAnomaly) = X(i,idxAnomaly);
    plot(XAnomalous,"r",LineWidth=3)
    hold off
end

xlabel("Time Step")

nexttile(1)
legend(["Input" "Anomalous"])


