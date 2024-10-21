import streamlit as st


# Help page
st.write("""
# O que é?
Este estudo propõe o uso de um modelo de aprendizado de máquina para a classificação e quantificação de tecidos musculares e conjuntivos em imagens histológicas de implantes tratados com plasma atmosférico a frio, visando superar as limitações da análise manual tradicional. A abordagem convencional, que utiliza pontos dispersos nas imagens e análise por especialistas, apresenta imprecisões devido ao número reduzido de amostras e à potencial ocultação de áreas relevantes. 
         
O modelo desenvolvido classifica pixels individualmente, permitindo uma análise mais detalhada e precisa. Os resultados mostraram um coeficiente $R^2$ de 0,908 na fase de validação com especialistas, demonstrando alta fidelidade das respostas. Além disso, o tempo de processamento foi reduzido significativamente, com cada imagem sendo analisada em menos de 0,15 segundos.

### Como usar?
1. Clique no botão "Carregar Arquivo" para selecionar uma imagem.
2. Aguarde a análise do modelo.
3. A classificação de cada será exibida na tela.
""")