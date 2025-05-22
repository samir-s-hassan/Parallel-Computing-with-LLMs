# Parallel Computing with LLMs

In this project, we test whether prompt design can push LLMs closer to expert-level 
parallel code. We compare a naïve "starter" prompt, a "student" prompt possessing a computer 
science student's knowledge base, a specialized "Palmieri" research‐persona prompt, and an 
"Emir" prompt demanding scalability. To measure real-world impact, we benchmark generated 
implementations on homework assignments—bank transactions, k-means clustering, and cuckoo 
hashing—across varying thread counts and data sizes. We also explore retrieval-augmented 
generation by feeding the model with textbook snippets on parallel and concurrent programming.

## final/
This directory contains the source code for each prompt/method of LLM code. There are five folders with each one corresponding to the specific method (prompt or RAG) used to generate LLM code.

Then, within each prompt/method folder, there are three folders corresponding to each assignment (Bank, Cuckoo, K-Means). You can find more information about these assignments on my Github. Each folder contains the LLM-generated code from ChatGPT, Claude, and Gemini for the assignment.

There's also a results folder which contains the execution times/other important information for each assignment's LLM-generated code. We used this for standardized benchmarking across all the different methods.

## Presentation
[Presentation.pdf](Presentation.pdf) - Contains our project presentation slides with analysis of the results and methodology. This was presented in-person to the rest of our class. Check this presentation out for a short summary of the project and our findings.

## Report
[Report.pdf](Report.pdf) - Comprehensive report detailing our results, methodology, and analysis of different prompt engineering approaches and the RAG. Check this report out for a detailed summary of the project and our findings.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Academic Attribution

This project was developed as part of a course assignment at Lehigh University. The work represents a collaborative effort between multiple team members. All contributors are listed in the [Presentation.pdf](Presentation.pdf) and the [Report.pdf](Report.pdf) file. 
