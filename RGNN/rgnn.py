import tensorflow as tf
import tensorflow_gnn as tfgnn
import numpy as np


class rgnn(tf.keras.Model):
    """
    A Recurrent Graph Neural Network model with a stateless graph creator
    to ensure compatibility with tf.data pipelines.
    """
    def __init__(self, adjacency_lists, num_iterations=10, hidden_dim=128, dropout_rate=0.4, **kwargs):
        super().__init__(**kwargs)
        self.L = num_iterations
        self.dropout_rate = dropout_rate
        self.hidden_dim = hidden_dim

        # Unpack the adjacency list into direct, explicit model attributes.
        self.NUM_CHECK_NODES = adjacency_lists['num_checks']
        self.NUM_VAR_NODES = adjacency_lists['num_vars']
        self.c_to_v_sources = adjacency_lists['c_to_v_sources']
        self.c_to_v_targets = adjacency_lists['c_to_v_targets']
        self.v_to_c_sources = adjacency_lists['v_to_c_sources']
        self.v_to_c_targets = adjacency_lists['v_to_c_targets']

        # Pre-calculate the number of edges as a static Python integer.
        self.num_c_to_v_edges = int(self.c_to_v_sources.shape[0])
        self.num_v_to_c_edges = int(self.v_to_c_sources.shape[0])

        self.check_node_input_transform = tf.keras.layers.Dense(self.hidden_dim)
        self.var_node_input_transform = tf.keras.layers.Dense(self.hidden_dim)
        self.recurrent_block = self._build_recurrent_message_block(self.hidden_dim)
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.final_predictor = tf.keras.layers.Dense(1, activation='sigmoid')


    def call(self, graph_tensor, training=False):
        graph_tensor = graph_tensor.replace_features(
            node_sets={
                'check_nodes': {'hidden_state': self.check_node_input_transform(
                    graph_tensor.node_sets['check_nodes']['hidden_state'])},
                'var_nodes': {'hidden_state': self.var_node_input_transform(
                    graph_tensor.node_sets['var_nodes']['hidden_state'])}
            })

        for _ in range(self.L):
            graph_tensor = self.recurrent_block(graph_tensor)

        var_features = graph_tensor.node_sets['var_nodes']['hidden_state']

        x = self.dropout(var_features, training=training)

        predictions = tf.squeeze(self.final_predictor(var_features), axis=-1)
        return predictions

    def _build_recurrent_message_block(self, hidden_dim):
        graph_update_layer = tfgnn.keras.layers.GraphUpdate(
            node_sets={
                'check_nodes': tfgnn.keras.layers.NodeSetUpdate(
                    edge_set_inputs={
                        'v_to_c': tfgnn.keras.layers.SimpleConv(
                            message_fn=tf.keras.layers.Dense(hidden_dim, activation='tanh'),
                            reduce_type='sum',
                            receiver_tag=tfgnn.TARGET
                        )
                    },
                    next_state=tfgnn.keras.layers.NextStateFromConcat(
                         tf.keras.layers.Dense(hidden_dim, activation='tanh')
                    )
                ),
                'var_nodes': tfgnn.keras.layers.NodeSetUpdate(
                    edge_set_inputs={
                        'c_to_v': tfgnn.keras.layers.SimpleConv(
                            message_fn=tf.keras.layers.Dense(hidden_dim, activation='tanh'),
                            reduce_type='sum',
                            receiver_tag=tfgnn.TARGET
                        )
                    },
                    next_state=tfgnn.keras.layers.NextStateFromConcat(
                        tf.keras.layers.Dense(hidden_dim, activation='relu')
                    )
                )
            }
        )
        return graph_update_layer

    @staticmethod
    def create_graph_tensor(syndrome_s, adj_data):
        """
        Creates a single GraphTensor. This is a static method to ensure it
        can be used in a tf.data.Dataset generator without state issues.
        """
        num_checks = adj_data['num_checks']
        num_vars = adj_data['num_vars']

        check_features_tensor = tf.cast(tf.expand_dims(syndrome_s, axis=1), tf.float32)
        check_features_tensor = tf.ensure_shape(check_features_tensor, [num_checks, 1])
        check_features = {'hidden_state': check_features_tensor}

        var_features_tensor = tf.zeros((num_vars, 1), dtype=tf.float32)
        var_features = {'hidden_state': var_features_tensor}

        return tfgnn.GraphTensor.from_pieces(
            node_sets={
                'check_nodes': tfgnn.NodeSet.from_fields(
                    sizes=[num_checks], features=check_features),
                'var_nodes': tfgnn.NodeSet.from_fields(
                    sizes=[num_vars], features=var_features),
            },
            edge_sets={
                'c_to_v': tfgnn.EdgeSet.from_fields(
                    sizes=[adj_data['c_to_v_sources'].shape[0]],
                    adjacency=tfgnn.Adjacency.from_indices(
                        source=('check_nodes', adj_data['c_to_v_sources']),
                        target=('var_nodes', adj_data['c_to_v_targets']))),
                'v_to_c': tfgnn.EdgeSet.from_fields(
                    sizes=[adj_data['v_to_c_sources'].shape[0]],
                    adjacency=tfgnn.Adjacency.from_indices(
                        source=('var_nodes', adj_data['v_to_c_sources']),
                        target=('check_nodes', adj_data['v_to_c_targets']))),
            }
        )

    def get_config(self):
        config = super().get_config()
        adj_lists_serializable = {
            'num_checks': self.NUM_CHECK_NODES,
            'num_vars': self.NUM_VAR_NODES,
            'c_to_v_sources': self.c_to_v_sources.numpy().tolist(),
            'c_to_v_targets': self.c_to_v_targets.numpy().tolist(),
            'v_to_c_sources': self.v_to_c_sources.numpy().tolist(),
            'v_to_c_targets': self.v_to_c_targets.numpy().tolist(),
        }
        config.update({
            "num_iterations": self.L,
            "adjacency_lists": adj_lists_serializable,
        })
        return config

    @classmethod
    def from_config(cls, config):
        adj_lists_config = config.pop("adjacency_lists")
        adj_lists_rebuilt = {
            'c_to_v_sources': tf.constant(adj_lists_config['c_to_v_sources'], dtype=tf.int64),
            'c_to_v_targets': tf.constant(adj_lists_config['c_to_v_targets'], dtype=tf.int64),
            'v_to_c_sources': tf.constant(adj_lists_config['v_to_c_sources'], dtype=tf.int64),
            'v_to_c_targets': tf.constant(adj_lists_config['v_to_c_targets'], dtype=tf.int64),
            'num_checks': adj_lists_config['num_checks'],
            'num_vars': adj_lists_config['num_vars']
        }
        return cls(adjacency_lists=adj_lists_rebuilt, **config)

