clc
clear all

% Türkçe karakterlerin hata vermemesi için 'preserve' kullanılıyor
opts = detectImportOptions('fetal_health.csv');
opts.VariableNamingRule = 'preserve';

% Veriyi oku
Data = readtable('fetal_health.csv');
disp(Data);

% Giriş ve çıkış değişkenlerini ayır
X = table2array(Data(:, 1:21)); % Giriş değişkenleri
y = table2array(Data(:, 22));   % Çıkış değişkeni

% Korelasyon matrisini görselleştir
corrMatrix = corr(X);
figure;
heatmap(corrMatrix, 'Title', 'Korelasyon Matrisi');

% Kutu Grafiği
figure;
boxplot(X, 'Labels', 1:21);
title('Kutu Grafiği');
xlabel('Değişkenler');
ylabel('Değerler');

% Korelasyon katsayılarını hesapla
corr_values = corr(X, y);

% Korelasyon katsayılarını görselleştir
figure;
bar(corr_values);
title('Çıkış Değişkeni (Fetal Sağlık) ile Girdi Değişkenleri Arasındaki Korelasyon');
xlabel('Girdi Değişkenleri');
ylabel('Korelasyon Katsayısı');
xticklabels(opts.VariableNames(1:21)); % Girdi değişkenlerinin adlarını x ekseninde göster

% En fazla etkileyen değişkenin indisini bul
[max_corr, max_corr_index] = max(abs(corr_values));
max_corr_variable = opts.VariableNames(max_corr_index);
disp(['En fazla etkileyen değişken: ' max_corr_variable{1} ', Korelasyon Katsayısı: ' num2str(max_corr)]);

% En yüksek ve en düşük korelasyona sahip girdi değişkenlerini belirle
[sorted_corr, sorted_index] = sort(abs(corr_values), 'descend');
onemli_sutunlar = sorted_index(1:6); % Önemli sütunlar: En yüksek korelasyona sahip ilk 5 sütun

% En az etkileyen değişkenin indisini bul
[min_corr, min_corr_index] = min(abs(corr_values));
min_corr_variable = opts.VariableNames(min_corr_index);
disp(['En az etkileyen değişken: ' min_corr_variable{1} ', Korelasyon Katsayısı: ' num2str(min_corr)]);

subplot_rows = 3;
subplot_cols = 2;
% Önemli sütunlar için yoğunluk grafiğini çizdir
figure;
for i = 1:length(onemli_sutunlar)
    subplot(subplot_rows, subplot_cols, i);
    histfit(X(:, onemli_sutunlar(i)));
    title(['Değişken ' num2str(onemli_sutunlar(i))]);
    xlabel('Değerler');
    ylabel('Frekans');
end

% Önemli sütunlar için scatter plotları oluştur
figure;
for i = 1:length(onemli_sutunlar)
    subplot(subplot_rows, subplot_cols, i);
    scatter(X(:, onemli_sutunlar(i)), y, 'Marker', '.');
    title(['Değişken ' num2str(onemli_sutunlar(i))]);
    xlabel(['Değişken ' num2str(onemli_sutunlar(i))]);
    ylabel('Fetal Sağlık');
end

% Normalizasyon (Min-Max Normalizasyonu)
minVals = min(X);
maxVals = max(X);
X_normalized = (X - minVals) ./ (maxVals - minVals);

% Eğitim ve test veri setlerini oluştur
cv = cvpartition(size(X, 1), 'Holdout', 0.3);
X_train = X_normalized(cv.training,:);
y_train = y(cv.training,:);
X_test = X_normalized(cv.test,:);
y_test = y(cv.test,:);

% Autoencoder modelini oluştur
hiddenSize = 3; % Gizli katman boyutu
numEpochs = 1000;  % Epoch sayısı
learningRate = 0.01; % Öğrenme oranı
L2Regularization = 0.001; % L2 regülarizasyon katsayısı

autoenc = trainAutoencoder(X_train', hiddenSize, ...
    'MaxEpochs', 1000, ...
    'L2WeightRegularization', 0.001);

% Encode edilmiş öznitelikleri elde et
encoded_X_train = encode(autoenc, X_train');
encoded_X_test = encode(autoenc, X_test');

% SVM sınıflandırıcıları oluştur
svm = templateSVM('Standardize',true,'KernelFunction','linear');

X_train_filled = fillmissing(encoded_X_train', 'linear');

% Fitcecoc modelini oluştur
classifier = fitcecoc(X_train_filled, y_train, 'Learners', svm);

% Eğitim veri seti için tahminleri yap
predicted_labels_train = predict(classifier, encoded_X_train');

% Test veri seti için tahminleri yap
predicted_labels_test = predict(classifier, encoded_X_test');

% Eğitim veri seti için confusion matrix hesapla
C_train = confusionmat(y_train, predicted_labels_train);

% Test veri seti için confusion matrix hesapla
C_test = confusionmat(y_test, predicted_labels_test);

% Eğitim veri seti için confusion matrix'i görselleştir
figure;
heatmap(C_train, 'ColorbarVisible', 'off', ...
    'XLabel', 'Tahmin Edilen Sınıf', 'YLabel', 'Gerçek Sınıf', ...
    'Title', 'Confusion Matrix (Eğitim Veri Seti)', ...
    'CellLabelColor', 'black', 'FontSize', 12);

% Test veri seti için confusion matrix'i görselleştir
figure;
heatmap(C_test, 'ColorbarVisible', 'off', ...
    'XLabel', 'Tahmin Edilen Sınıf', 'YLabel', 'Gerçek Sınıf', ...
    'Title', 'Confusion Matrix (Test Veri Seti)', ...
    'CellLabelColor', 'black', 'FontSize', 12);

% Eğitim seti accuracy hesapla
accuracy_train = sum(diag(C_train)) / sum(C_train(:));

% Test seti accuracy hesapla
accuracy_test = sum(diag(C_test)) / sum(C_test(:));

fprintf('Eğitim seti accuracy: %.2f%%\n', accuracy_train * 100);
fprintf('Test seti accuracy: %.2f%%\n', accuracy_test*100);

% Eğitim seti için MSE hesapla
mse_train = mean((predicted_labels_train - y_train).^2);

% Test seti için MSE hesapla
mse_test = mean((predicted_labels_test - y_test).^2);

fprintf('Eğitim seti MSE: %.2f\n', mse_train);
fprintf('Test seti MSE: %.2f\n', mse_test);
