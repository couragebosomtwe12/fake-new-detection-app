<?php require_once 'articles_data.php'; ?>
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>The Daily Truth Exposed - Breaking News</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: Georgia, 'Times New Roman', serif;
            background-color: #f5f5f0;
            color: #1a1a1a;
        }
        .container { max-width: 900px; margin: 0 auto; padding: 20px; }
        .header {
            background: linear-gradient(135deg, #8B0000 0%, #5C0000 100%);
            color: white;
            padding: 30px 20px;
            text-align: center;
            border-radius: 8px;
            margin-bottom: 30px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        }
        .header h1 { font-size: 2.2em; letter-spacing: 2px; text-transform: uppercase; }
        .header p { font-size: 1.1em; margin-top: 10px; opacity: 0.9; }
        .breaking-bar {
            background-color: #8B0000;
            color: white;
            text-align: center;
            padding: 8px;
            font-weight: bold;
            font-size: 0.9em;
            letter-spacing: 1px;
            margin-bottom: 20px;
            border-radius: 4px;
        }
        .article {
            background-color: white;
            padding: 25px;
            margin-bottom: 25px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            border-left: 5px solid #8B0000;
        }
        .headline {
            color: #8B0000;
            font-size: 1.5em;
            font-weight: bold;
            margin-bottom: 8px;
            line-height: 1.3;
        }
        .headline a {
            color: inherit;
            text-decoration: none;
        }
        .headline a:hover { text-decoration: underline; }
        .meta {
            color: #666;
            font-size: 0.85em;
            margin-bottom: 15px;
            border-bottom: 1px solid #eee;
            padding-bottom: 10px;
        }
        .meta .author { font-weight: bold; color: #8B0000; }
        .body { line-height: 1.8; color: #333; font-size: 1.05em; }
        .body p { margin-bottom: 15px; }
        .read-more {
            display: inline-block;
            margin-top: 10px;
            color: #8B0000;
            font-weight: bold;
            text-decoration: none;
        }
        .read-more:hover { text-decoration: underline; }
        .share-bar {
            margin-top: 15px;
            padding: 10px;
            background-color: #fff3cd;
            border: 1px solid #ffc107;
            border-radius: 4px;
            font-size: 0.9em;
            text-align: center;
            color: #856404;
            font-weight: bold;
        }
        .footer {
            text-align: center;
            padding: 30px;
            color: #666;
            font-size: 0.85em;
            border-top: 1px solid #ddd;
            margin-top: 30px;
        }
        .disclaimer {
            background-color: #d4edda;
            border: 2px solid #28a745;
            color: #155724;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            margin-top: 30px;
            font-size: 0.9em;
        }
        .disclaimer strong { display: block; margin-bottom: 5px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>The Daily Truth Exposed</h1>
            <p>News They Don't Want You To Know</p>
        </div>

        <div class="breaking-bar">
            &#9888;&#65039; BREAKING: These stories are being suppressed by the mainstream media &#9888;&#65039;
        </div>

        <?php foreach ($articles as $article): ?>
        <article class="article" id="article-<?= $article['id'] ?>">
            <div class="headline">
                <a href="article.php?id=<?= $article['id'] ?>"><?= htmlspecialchars($article['headline']) ?></a>
            </div>
            <div class="meta">
                <span class="author"><?= htmlspecialchars($article['author']) ?></span>
                &bull; Published: <?= htmlspecialchars($article['date']) ?>
            </div>
            <div class="body">
                <?php
                $paragraphs = explode("\n\n", $article['body']);
                // Show only the first 2 paragraphs on the homepage
                foreach (array_slice($paragraphs, 0, 2) as $para): ?>
                <p><?= htmlspecialchars($para) ?></p>
                <?php endforeach; ?>
            </div>
            <a class="read-more" href="article.php?id=<?= $article['id'] ?>">Read Full Article &rarr;</a>
            <div class="share-bar">
                &#128226; SHARE THIS BEFORE IT GETS DELETED &#128226;
            </div>
        </article>
        <?php endforeach; ?>

        <div class="disclaimer">
            <strong>&#9888;&#65039; TEST PURPOSE ONLY &#9888;&#65039;</strong>
            This website contains deliberately fake news articles designed to test fake news detection systems.
            All content here is fabricated and should not be taken as real news or medical advice.
        </div>

        <div class="footer">
            <p>&copy; 2026 The Daily Truth Exposed. All rights reserved.</p>
        </div>
    </div>
</body>
</html>