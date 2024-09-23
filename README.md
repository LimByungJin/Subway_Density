# Subway_Density
https://doi.org/10.3390/electronics12040917
CGAN을 이용하여 변조된 지하철 이용자 데이터로부터 지하철 사용자 분포를 생성 (Generate subway station user density from the noisy data with CGAN)

# Data
data direcotry의 clean_raw_data.txt는 변조되지 않은 원본 데이터로 각 행의 첫 2열은 각 0 ~ 7, 0 ~ 17로 요일과 시간대를 나타내며 그 이후 114개의 역에대한 이용자의 수를 나타내는 데이터 파일.

uniform_perturbed_0.1.txt는 원본으로부터 변조된 이용자의 수이며 요일과 시간 부분이 삭제된 데이터 파일.
