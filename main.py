import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

import datasets as dset
from models import Generator, Discriminator, weights_init

cudnn.benchmark = True

input_nc = output_nc = 3

num_epochs = 200
batch_size = 64
η = 0.001
β1 = 0.5
λ = 100

num_workers = 4

tf = transforms.Compose([
    transforms.Scale(256),
    transforms.ToTensor(),
])

print('===> Load datasets')
train_dataset = dset.ImageFolderDataset(root='./datasets/facades/train',
                                        transform=tf)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           num_workers=num_workers,
                                           pin_memory=True,
                                           shuffle=True)

print('===> Build model')
generator = Generator(3, 3, 64)
discriminator = Discriminator(3, 3, 64)
generator.apply(weights_init)
discriminator.apply(weights_init)
generator.cuda()
discriminator.cuda()

print(generator)
print(discriminator)

print('===> Build loss and optimizers')
criterion = nn.BCELoss().cuda()
criterion_l1 = nn.L1Loss().cuda()

g_optimizer = torch.optim.Adam(
    generator.parameters(), lr=η, betas=(β1, 0.999))
d_optimizer = torch.optim.Adam(
    discriminator.parameters(), lr=η, betas=(β1, 0.999))

print('===> Train model')

real_a = Variable(torch.FloatTensor(batch_size, input_nc, 256, 256).cuda())
real_b = Variable(torch.FloatTensor(batch_size, output_nc, 256, 256).cuda())
label = Variable(torch.FloatTensor(batch_size).cuda())
real_label, fake_label = 1, 0

for epoch in range(num_epochs):
    for i, (img_a, img_b) in enumerate(train_loader):
        real_a.data.resize_(img_a.size()).copy_(img_a)
        real_b.data.resize_(img_a.size()).copy_(img_b)

        ''' discriminator '''
        d_optimizer.zero_grad()

        # Train with real
        real_ab = torch.cat((real_a, real_b), 1)
        output = discriminator(real_ab)
        label.data.resize_(output.size()).fill_(real_label)
        real_loss = criterion(output, label)
        real_loss.backward()
        d_x_y = output.data.mean()

        # Train with fake
        fake_b = generator(real_a).detach()
        fake_ab = torch.cat((real_a, fake_b), 1)
        output = discriminator(fake_ab)
        label.data.resize_(output.size()).fill_(fake_label)
        fake_loss = criterion(output, label)
        fake_loss.backward()
        d_x_gx = output.data.mean()

        d_loss = (real_loss + fake_loss) / 2.0

        d_optimizer.step()

        ''' generator '''
        generator.zero_grad()

        output = discriminator(fake_ab)
        label.data.resize_(output.size()).fill_(real_label)
        g_loss = criterion(output, label) + λ * criterion_l1(fake_b, real_b)
        g_loss.backward()
        d_x_gx_2 = output.data.mean()

        g_optimizer.step()

        if (i + 1) % 2 == 0:
            print('Epoch [%d/%d](%02d/%02d) => '
                  % (epoch + 1, num_epochs,
                     i + 1, len(train_dataset) // batch_size), end=' ')
            print('Loss_D: {:.4f} Loss_G: {:.4f}'
                  ' D(x): {:.4f} D(G(z)): {:.4f}/{:.4f}'.format(
                      d_loss.data[0], g_loss.data[0], d_x_y, d_x_gx, d_x_gx_2))

    torchvision.utils.save_image(
        fake_b.data, './fake_samples_epoch%d.png' % (epoch + 1))

torch.save(generator.state_dict(), './generator.pkl')
torch.save(discriminator.state_dict(), './discriminator.pkl')
