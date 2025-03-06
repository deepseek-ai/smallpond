Architecture
============

smallpond uses a DAG-based execution model with lazy evaluation:

1. Operations build a logical plan as a directed acyclic graph (DAG)
2. Execution is triggered only when an action is called (write, compute, etc.)
3. Ray distributes tasks across workers, with each worker running its own DuckDB instance
4. Backend storage supported is 3FS, while local filesystem can also be used for development and testing

.. mermaid::
   :align: center
   :caption: smallpond Architecture

   flowchart TD
       %% Main Flow subgraph at the top
       subgraph MF[Main Flow]
           direction TB
           User([User]):::userFlow
           Code[User Code]:::userFlow
           DAG[Logical Plan DAG]:::userFlow
           Partitions[(Partitioned Data)]
           
           User --> |"1 Creates DataFrame<br>operations"| Code
           Code --> |"2 Builds"| DAG
           DAG --> |"3 Optimizes & manual<br>partitions data"| Partitions
           User --> |"6 Triggers execution<br>(write_parquet, compute)"| DAG
       end

       %% Distributed Execution subgraph
       subgraph DE[Distributed Execution]
           direction TB
           RayCluster[Ray Cluster]:::execution
           
           %% Workers level
           Worker1[Worker 1]:::execution
           Worker2[Worker 2]:::execution
           Worker3[Worker 3]:::execution
           
           %% DuckDB level
           DuckDB1[DuckDB Instance]:::execution
           DuckDB2[DuckDB Instance]:::execution
           DuckDB3[DuckDB Instance]:::execution

           %% Internal connections
           RayCluster --> Worker1
           RayCluster --> Worker2
           RayCluster --> Worker3
           
           Worker1 --> |"5a Processes<br>partition"| DuckDB1
           Worker2 --> |"5b Processes<br>partition"| DuckDB2
           Worker3 --> |"5c Processes<br>partition"| DuckDB3
       end

       %% Bottom row with Storage and Results side by side
       subgraph Bottom[ ]
           direction LR
           subgraph SO[Storage Options]
               direction LR
               Storage[(Storage Layer)]:::storage
               3FS[3FS]:::storage
               LocalFS[Local FS]:::storage
               
               Storage --> 3FS
               Storage --> LocalFS
           end

           subgraph RC[Results Collection]
               direction LR
               Results[Results Collection]:::userFlow
           end
       end

       %% Connect subgraphs
       Partitions --> |"4 Distributes tasks<br>via Ray"| RayCluster
       Partitions -.-> Storage

       %% Storage connections with dotted lines
       DuckDB1 -.-> |"Reads/Writes"| Storage
       DuckDB2 -.-> |"Reads/Writes"| Storage
       DuckDB3 -.-> |"Reads/Writes"| Storage

       %% Results collection
       DuckDB1 --> |"7a Produces"| Results
       DuckDB2 --> |"7b Produces"| Results
       DuckDB3 --> |"7c Produces"| Results
       Results --> |"8 Returns"| User

       %% Styling
       classDef userFlow fill:#ff69b4,stroke:#333333,stroke-width:2px,color:#ffffff
       classDef execution fill:#4CAF50,stroke:#333333,stroke-width:1px,color:#ffffff
       classDef storage fill:#2196F3,stroke:#333333,stroke-width:1px,color:#ffffff
       classDef default color:#ffffff
