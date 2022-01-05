
우이ㅣ히히아아아아

우헤헤ㅔ헤헤헤헤

Xiang Zhang, Junbo Zhao, & Yann LeCun. (2016). Character-level Convolutional Networks for Text Classification.

논문 정리는 마크다운으로 해야지!
아 코드 상태 맘에 안들어
다음부터는 colab에서 바로 push해야겠어

아쉬운점은 pytorch로 구현하지 않고 keras로 구현했다는 점!
pytorch에 

TIL
- OOV_ token
- 오프셋 상수

파라미터 계산


느낀점
CNN으로도 텍스트 학습 가능하구나~ 역시 멀티 모델러가 되야해!

**그리고 Conv: h(y)에 관해서 헷갈림**


논문에서 처음에는 conv 로 h(y)를 계산했는데 f(x)랑 g(y\*d -x+c)를 곱하고 x에 대해 더하는거!
근데 뒤에는 h(y)가 max pooling function으로 나옴.
물론 conv 하는 과정에서 max pooling 과정이 들어가긴 하지만 그래도 수식이 저래도 되는건가?

$$ h(y) = \sum_{x=1}^{k} f(x) \cdot g(y\cdot d-x+c)$$ -(1)
$$h(y) = \max_{x=1}^{k} g(y \cdot d -x +c)$$ -(2)

(1)과 (2)가 같다면, f(x)가 굳이 필요없는 거 아닌가? cnn을 하는데 커널이 필요없어지는건데 사실 말도 안됨.
혹시 1번의 h(y)가 끝나고 다음과정에서 2번의 h(y)가 등장하는건가? 
(Conv 끝내고 풀링하는것처럼, 근데 다른 과정인데 왜 기호를 똑같이 썼지??)
이건 discussion해봐야겠다
