add_executable(ninfer-yarn
  cli/main.cpp
  cli/options.cpp)
ninfer_internal_includes(ninfer-yarn)
target_link_libraries(ninfer-yarn PRIVATE
  ninfer_engine
  ninfer_product_logging
  ninfer_product_prompt_input)

add_executable(ninfer-yarn-serve serve/main.cpp)
ninfer_internal_includes(ninfer-yarn-serve)
target_link_libraries(ninfer-yarn-serve PRIVATE ninfer_serve ninfer_product_logging)

add_executable(ninfer-yarn-perplexity
  perplexity/main.cpp
  perplexity/corpus.cpp
  perplexity/evaluation.cpp)
ninfer_internal_includes(ninfer-yarn-perplexity)
target_link_libraries(ninfer-yarn-perplexity PRIVATE
  ninfer_engine ninfer_product_logging ninfer::json)
