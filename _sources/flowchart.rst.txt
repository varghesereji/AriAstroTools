Module Flowchart
================

This flowchart shows the main workflow of the package:

.. mermaid::

   graph TD
      Start[Start] --> Input[User enters CLI command]
      Input --> Parse[command-line arguments]
      Parse --> Decision{Which command?}
      Decision -->|combine mean,median,biweight| CombineFunc[Run combine function]
      Decision -->|operation +,-,*,/| OperFunc[Run operation function]
      
      CombineFunc -->|--files File1 File2 ... --output opfile| FileLoad[Load input files]
      FileLoad --> Combine[Combine based on method]
      
      OperFunc -->|--files File1 File2 ... --output opfile| FileLoad2[Load input files]
      FileLoad2 --> Operation[Operate based on the input]
      Combine --> Outputfile[Save the output file]
      Operation --> Outputfile
      Outputfile --> End[End]


      
       %% clickable nodes
       click CombineFunc "source/operations.html" "Go to Usage page"
       click OperFunc "source/operations.html" "Go to Usage page"
       click C "config.html" "Go to Config page"
