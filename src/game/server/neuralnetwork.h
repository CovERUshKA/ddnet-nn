#include <vector>

enum LayerType
{
	LAYER_INPUT,
	LAYER_HIDDEN,
	LAYER_OUTPUT
};

class Layer
{
	int type;
	std::vector<float> neurons;
};

	class NeuralNetwork
{
    //std::vector<> layers;

    bool Initialize();

    bool AddLayer(int count_units, int type);

    bool nn();
};