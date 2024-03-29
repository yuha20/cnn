#ifndef NETWORK2_H
#define NETWORK2_H

#include <initializer_list>
#include <vector>
#include <tuple>
#include <memory>
#include <atomic>
#include <yannpp/common/cpphelpers.h>
#include <yannpp/common/log.h>
#include <yannpp/optimizer/optimizer.h>
#include <yannpp/layers/fullyconnectedlayer.h>
#include <yannpp/network/activator.h>

namespace yannpp {
    template<typename T>
    class network2_t {
    public:
        using data_type = T;
        using t_d = array3d_t<data_type>;
        using training_data = std::vector<std::tuple<t_d, t_d>>;
        using layer_type = std::shared_ptr<layer_base_t<data_type>>;

    public:
        network2_t(std::initializer_list<layer_type> layers):
            layers_(layers)
        {}

        network2_t(std::vector<layer_type> &&layers):
            layers_(std::move(layers))
        {}

    public:
        void init_layers() {
            for (auto &l: layers_) {
                l->init();
            }
        }

        void train(network2_t::training_data const &data,
                   optimizer_t<data_type> const &optimizer,
                   size_t epochs,
                   size_t minibatch_size) {
            log("Training using %d inputs", data.size());
            // big chunk of data is used for training while
            // small chunk - for validation after some epochs
            const size_t training_size = 5 * data.size() / 6;
            std::vector<size_t> eval_indices(data.size() - training_size);
            // generate indices from 1 to the number of inputs
            std::iota(eval_indices.begin(), eval_indices.end(), training_size);

            for (size_t e = 0; e < epochs; e++) {
                auto indices_batches = batch_indices(training_size, minibatch_size);
                const size_t batches_size = indices_batches.size();

                for (size_t b = 0; b < batches_size; b++) {
                    auto start = system_clock::now();
                    update_mini_batch_parallel(data, indices_batches[b], optimizer);
                    auto end = system_clock::now();
                    auto cost = std::chrono::duration<double, std::micro>(end - start).count();
                    if (b % (batches_size/4) == 0) { log("Processed batch %d out of %d cost %f" , b, batches_size,cost); }
                }

                auto result = evaluate(data, eval_indices);
                log("Epoch %d: %d / %d", e, result, eval_indices.size());
            }
            auto start = system_clock::now();
            auto result = evaluate(data, eval_indices);
            auto end1 = system_clock::now();
            auto result1 = evaluate_parallel(data, eval_indices);
            auto end2 = system_clock::now();
            auto result2 = evaluate_SIMD(data, eval_indices);
            auto end3 = system_clock::now();
            auto cost = std::chrono::duration<double, std::micro>(end1 - start).count();
            auto cost1 = std::chrono::duration<double, std::micro>(end2 - end1).count();
            auto cost2 = std::chrono::duration<double, std::micro>(end3 - end2).count();
            log("cost: %f / %f / %f", cost, cost1, cost2);
            log("End result: %d / %d", result, eval_indices.size());
            log("End result: %d / %d", result1, eval_indices.size());
            log("End result: %d / %d", result2, eval_indices.size());
        }

        // feeds input a to the network and returns output
        t_d feedforward(t_d const &a) {
            array3d_t<network2_t::data_type> input(a);
            for (auto &layer: layers_) {
                input = layer->feedforward(std::move(input));
            }
            return input;
        }
        // feeds input a to the network and returns output
        t_d feedforward_parallel(t_d const &a) {
            const size_t layers_size = layers_.size();
            array3d_t<network2_t::data_type> input(a);
            std::vector< array3d_t <network2_t::data_type> > inputs;
            std::vector< std::deque<array3d_t<T>> > patches(layers_size);
            std::vector <array3d_t<T> > outputs(layers_size);
            // feedforward_parallel input
            for (size_t i = 0; i < layers_size; i++) {
                inputs.push_back(input);
                input = layers_[i]->feedforward_parallel(inputs[i], outputs[i]);
            }
            return input;
        }
#define INPUT(i) std::get<0>(data[i])
#define RESULT(i) std::get<1>(data[i])

        // evaluates number of correctly classified inputs (validation data)
        size_t evaluate(training_data const &data, std::vector<size_t> const &indices) {
            size_t count = 0;
            for (auto i: indices) {
                network2_t::t_d result = feedforward(INPUT(i));
                assert(result.size() == RESULT(i).size());
                if (argmax1d(result) == argmax1d(RESULT(i))) { count++; }
            }
            return count;
        }
        size_t evaluate_SIMD(training_data const &data, std::vector<size_t> const &indices) {
            size_t count = 0;
            for (auto i: indices) {
                network2_t::t_d result = feedforward_parallel(INPUT(i));
                assert(result.size() == RESULT(i).size());
                if (argmax1d(result) == argmax1d(RESULT(i))) { count++; }
            }

            return count;
        }
        size_t evaluate_parallel(training_data const &data, std::vector<size_t> const &indices) {
            std::atomic<int> count{0};
//            std::atomic<int>
            tbb::parallel_for(tbb::blocked_range<size_t>(0, indices.size(),20), [&](const tbb::blocked_range <size_t> &r) {
                for ( int i=r.begin();i!=r.end();i++) {
                    network2_t::t_d result=feedforward_parallel(INPUT(indices[i]));
                    assert(result.size() == RESULT(i).size());
                    if (argmax1d(result) == argmax1d(RESULT(indices[i]))) {
                        count++;
                    }
                }
            });

            return count;
        }
    private:
        // updates network weights and biases using one
        // iteration of gradient descent using mini_batch of inputs and outputs
        void update_mini_batch(training_data const &data,
                               std::vector<size_t> const &indices,
                               optimizer_t<network2_t::data_type> const &strategy) {
            for (auto i: indices) {
                backpropagate(INPUT(i), RESULT(i));
            }

            for (auto &layer: layers_) {
                layer->optimize(strategy);
            }
        }
        void update_mini_batch_parallel(training_data const &data,
                               std::vector<size_t> const &indices,
                               optimizer_t<network2_t::data_type> const &strategy) {

          for (auto i: indices) {
//            backpropagate_parallel(INPUT(i), RESULT(i));
            backpropagate(INPUT(i), RESULT(i));
          }

          for (auto &layer: layers_) {
            layer->optimize(strategy);
          }
        }

        // runs a loop of propagation of inputs and backpropagation of errors
        // back to the beginning with weights and biases updates as a result
        void backpropagate(t_d const &x, t_d const &result) {
            const size_t layers_size = layers_.size();
            array3d_t<network2_t::data_type> input(x);


            // feedforward_parallel input
            for (size_t i = 0; i < layers_size; i++) {

                input = layers_[i]->feedforward(std::move(input));
            }

            // backpropagate error
            array3d_t<network2_t::data_type> error(result);
            for (size_t i = layers_size; i-- > 0;) {
                error = layers_[i]->backpropagate(std::move(error));
            }
        }
        void backpropagate_parallel(t_d const &x,t_d const &result) {

            const size_t layers_size = layers_.size();
            array3d_t<network2_t::data_type> input(x);
            std::vector< array3d_t <network2_t::data_type> > inputs;
            std::vector< std::deque<array3d_t<T>> > patches(layers_size);
            std::vector <array3d_t<T> > outputs(layers_size);
            // feedforward_parallel input
            for (size_t i = 0; i < layers_size; i++) {
              inputs.push_back(input);
              input = layers_[i]->feedforward_parallel(inputs[i], outputs[i]);
            }

            // backpropagate error
            array3d_t<network2_t::data_type> error(result);
            for (size_t i = layers_size; i-- > 0;) {
              error = layers_[i]->backpropagate(std::move(error),inputs[i],outputs[i]);
            }
        }
    private:
        std::vector<std::shared_ptr<layer_base_t<data_type>>> layers_;
    };
}

#endif // NETWORK2_H
