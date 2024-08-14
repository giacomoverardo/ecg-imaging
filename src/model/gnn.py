from tensorflow import keras
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, LayerNormalization
from spektral.layers import GCNConv, DiffPool, GATConv
import tensorflow as tf
from typing import Dict, List, Tuple

class Uniform(Model):
    """Model that retuns a uniform 1s vector when called
       The returned shape of the call method is (batch_size,num_features)
       which are obtained from the input inputs["signal"]

    Args:
        Model (_type_): _description_
    """
    def __init__(self, name, *args, **kwargs):
        super().__init__()
        self.gnn_name = name

    def call(self, inputs, training=None, mask=None):
        x = inputs["signal"]
        batch_size = tf.shape(x)[0]
        num_features = tf.shape(x)[1]
        return tf.ones((batch_size,num_features))
    
class GCNModel(Model):
    def __init__(
        self,
        name,
        hidden_nodes: List[int],
        dense_hidden_nodes: List[int],
        final_dense_nodes: int,
    ):
        super().__init__()
        self.gnn_name = name
        # Ensure the layer sizes are correct
        assert len(hidden_nodes) == len(dense_hidden_nodes)

        self.num_layers = len(hidden_nodes)
        self.gcn_layers = []
        self.dense_layers = []
        self.pool_layers = []

        for i in range(self.num_layers):
            self.gcn_layers.append(GCNConv(hidden_nodes[i]))
            self.dense_layers.append(Dense(dense_hidden_nodes[i], activation="relu"))

        self.final_dense = Dense(
            final_dense_nodes, "linear"
        )  # Adjust the output dimension

    def call(self, inputs, training=True):
        x = inputs["signal"]
        a = inputs["adj_matrix"]
        for j in range(self.num_layers):
            x = self.gcn_layers[j]([x, a])
            x = self.dense_layers[j](x)
        out = self.final_dense(x)
        out = tf.squeeze(out, axis=-1)
        out = tf.nn.sigmoid(out)
        return out

class GATModel(Model):
    """
    A GAT-based model for processing graph-structured data.
    """
    def __init__(
        self,
        name,
        embedding_nodes:int,
        channels_list: List[int],
        attn_heads_list: List[int],
        dense_nodes_list: List[int],
        final_dense_nodes: int,
        activation: str = 'relu'
    ):
        super().__init__()
        self.gnn_name = name
        assert len(channels_list) == len(attn_heads_list) == len(dense_nodes_list), "Channels and attention heads and dense_nodes_list must have the same length"

        # Define embedding layer to assign correct shape to input
        self.embedding = Dense(embedding_nodes, activation="relu") 
        # Define GAT blocks layers and hyper-parameters
        self.num_blocks = len(channels_list)
        self.gat_layers = []
        self.dense_layers = []
        self.layer_norm_layers_1 = []
        self.layer_norm_layers_2 = []
        self.activation = activation
        # Initialize GAT-based blocks
        for i in range(self.num_blocks):
            self.gat_layers.append(GATConv(channels=channels_list[i], attn_heads=attn_heads_list[i], activation=activation))
            self.dense_layers.append(Dense(units=dense_nodes_list[i], activation=activation))
            self.layer_norm_layers_1.append(LayerNormalization())
            self.layer_norm_layers_2.append(LayerNormalization())
        # Define final fully-connected layer to reshape output
        self.final_dense = Dense(final_dense_nodes, activation='linear')

    def call(self, inputs, training=True):
        # Extract node features and adjacency matrix from inputs
        x = inputs["signal"]
        a = inputs["adj_matrix"]
        # Apply embedding layer
        x = self.embedding(x)
        # Apply GAT-based block layers sequentially
        for j in range(self.num_blocks):
            out_gat = self.gat_layers[j]([x, a]) # Apply Gat layer
            out_sum_1 = out_gat + x # Add output from gat layer and original input x
            out_norm_1 = self.layer_norm_layers_1[j](out_sum_1) # Apply first layer normalization block
            out_dense = self.dense_layers[j](out_norm_1)    # Apply fully connected layer
            out_sum_2 = out_dense + out_norm_1              # Add output from dense and output from first layer normalization
            out_norm_2 = self.layer_norm_layers_2[j](out_sum_2) # Apply second layer normalization block
            x = out_norm_2 # Set input from next block
        # Apply final dense layer and sigmoid activation
        out = self.final_dense(x)
        out = tf.squeeze(out, axis=-1)
        out = tf.nn.sigmoid(out)
        return out
    
if __name__ == "__main__":
    pass