css = '''
<style>
    .main {
      background-color: #030C1A;
    }
    
    [data-testid="stMarkdownContainer"].st-emotion-cache-eqffof.e1nzilvr5 {
      color: #fff;
    }

    [data-testid="stChatMessage"].stChatMessage.st-emotion-cache-4oy321.eeusbqq4 {

      align-items: center;
      padding: 12px 16px;
      gap: 10px;
      background: #121F33;
      border-radius: 24px;
    }

    [data-testid="stChatMessage"].stChatMessage.st-emotion-cache-1c7y2kd.eeusbqq4 {
      
      align-items: center;
      padding: 12px 16px;
      gap: 10px;
      background: #1D74F5;
      border-radius: 24px;
    }

    
    </style>
'''




assistant_template = '''
<div class="chat-message asst">
    <div class="message">{{msg}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">   
    <div class="message">{{msg}}</div>
</div>
'''