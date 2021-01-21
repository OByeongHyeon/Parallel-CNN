# Parallel Convolutional Neural Networks

아래의 두 연산 즉, CNN 을 구성하는 Convolutional & Max-pooling layer 의 수행 연산을 병렬로 처리하기 위해 multi-process & multi-thread 기술을 사용

- [multi-process Parallel CNN](Process_based%20Paralell%20CNN) 

- [multi-thread Parallel CNN](Thread_based%20Paralell%20CNN)



### Convolution(합성곱) 연산
<img src="https://user-images.githubusercontent.com/67847920/105355071-de25b500-5c34-11eb-9792-5c60ecc5ce38.png" width="450"/>



### Max-pooling 연산
<img src="https://user-images.githubusercontent.com/67847920/105355639-c1d64800-5c35-11eb-87ed-8d7757485b7f.png" width="300"/>

<br>

목표
--

1. 주어진 Input Matrix에 대하여 Filter를 이동해 가면서 순서대로 합성곱 연산을 하는 것이 아닌
    각 위치(3*3 단위)에서의 합성곱 연산을 병렬로 수행하여 Output Matrix 를 생성하는 것이 목표
<br>
2. 마찬가지로 앞 과정에서 만들어진 Output Matrix 에 대하여 각 위치(2*2 단위)에서의 Max-pooling 연산을 병렬로 수행하여 최종적인 결과 Matrix 를 생성하는 것이 목표