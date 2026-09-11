<?php
// Shared article data — same as index.php
require_once 'articles_data.php';

$id = isset($_GET['id']) ? (int)$_GET['id'] : 0;
$article = null;
foreach ($articles as $a) {
    if ($a['id'] === $id) {
        $article = $a;
        break;
    }
}

if (!$article) {
    http_response_code(404);
    $page_title = "Article Not Found";
} else {
    $page_title = $article['headline'];
}
?>
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title><?= htmlspecialchars($page_title) ?> - The Daily Truth Exposed</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: Georgia, 'Times New Roman', serif;
            background-color: #f5f5f0;
            color: #1a1a1a;
        }
        .container { max-width: 800px; margin: 0 auto; padding: 20px; }
        .header {
            background: linear-gradient(135deg, #8B0000 0%, #5C0000 100%);
            color: white;
            padding: 20px;
            text-align: center;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        .header h1 { font-size: 1.6em; }
        .header a { color: #ffd700; text-decoration: none; }
        .article {
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .headline {
            color: #8B0000;
            font-size: 1.8em;
            font-weight: bold;
            margin-bottom: 10px;
            line-height: 1.3;
        }
        .meta {
            color: #666;
            font-size: 0.85em;
            margin-bottom: 20px;
            border-bottom: 1px solid #eee;
            padding-bottom: 12px;
        }
        .meta .author { font-weight: bold; color: #8B0000; }
        .body { line-height: 1.8; color: #333; font-size: 1.05em; }
        .body p { margin-bottom: 15px; }
        .share-bar {
            margin-top: 20px;
            padding: 12px;
            background-color: #fff3cd;
            border: 1px solid #ffc107;
            border-radius: 4px;
            text-align: center;
            color: #856404;
            font-weight: bold;
        }
        .back-link {
            display: inline-block;
            margin-top: 20px;
            color: #8B0000;
            text-decoration: none;
        }
        .error-box {
            background: white;
            padding: 40px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .error-box h2 { color: #8B0000; margin-bottom: 15px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1><a href="index.php" style="color:white;">The Daily Truth Exposed</a></h1>
            <p>News They Don't Want You To Know</p>
        </div>

        <?php if ($article): ?>
        <article class="article">
            <div class="headline"><?= htmlspecialchars($article['headline']) ?></div>
            <div class="meta">
                <span class="author"><?= htmlspecialchars($article['author']) ?></span>
                &bull; Published: <?= htmlspecialchars($article['date']) ?>
            </div>
            <div class="body">
                <?php foreach (explode("\n\n", $article['body']) as $para): ?>
                <p><?= htmlspecialchars($para) ?></p>
                <?php endforeach; ?>
            </div>
            <div class="share-bar">
                &#128226; SHARE THIS BEFORE IT GETS DELETED &#128226;
            </div>
        </article>
        <?php else: ?>
        <div class="error-box">
            <h2>Article Not Found</h2>
            <p>The article you're looking for has been removed (probably by the government).</p>
            <a href="index.php" class="back-link">&larr; Back to home</a>
        </div>
        <?php endif; ?>
    </div>
</body>
</html>